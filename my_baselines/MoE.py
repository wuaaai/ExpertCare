import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dims,                             
        num_domains,                   
        num_tasks,                   
        hidden_dim,
        dropout_rate=0.1
    ):
        super(MoE, self).__init__()
        self.num_domains = num_domains
        self.num_tasks = num_tasks
        self.output_dims = output_dims
        self.max_output_dim = max(output_dims)

        self.domain_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, self.max_output_dim),
                nn.LayerNorm(self.max_output_dim)
            ) for _ in range(num_domains)
        ])

        self.task_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.max_output_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dims[i])
            ) for i in range(num_tasks)
        ])

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)                      
        )

        self.min_other_weight = 0.05
        self.learnable_main_weight = nn.Parameter(torch.tensor(0.9), requires_grad=True)

    def forward(self, x, domain_ids, task=None):
        device = x.device
        batch_size = x.size(0)

        domain_outputs = torch.stack([expert(x) for expert in self.domain_experts], dim=1)                                    

        gate_scores = self.gating_network(x)                    
        gate_probs = torch.sigmoid(gate_scores)                    

        expert_weights = torch.zeros((batch_size, self.num_domains), device=device)

        for i in range(batch_size):

            main_ids = domain_ids[i]
            if isinstance(main_ids, torch.Tensor):
                main_ids = main_ids.tolist()
            if not isinstance(main_ids, (list, tuple)):
                main_ids = [main_ids]            

            main_weights = (self.learnable_main_weight * gate_probs[i, main_ids])                          
            main_weights = main_weights / main_weights.sum()       

            for idx, mid in enumerate(main_ids):
                expert_weights[i, mid] = main_weights[idx]

            remaining_weight = max(0.0, 1.0 - expert_weights[i, main_ids].sum().item())
            num_other = self.num_domains - len(main_ids)
            if num_other > 0:
                other_ids = [j for j in range(self.num_domains) if j not in main_ids]
                for j in other_ids:
                    expert_weights[i, j] = remaining_weight / num_other

        domain_output = torch.sum(domain_outputs * expert_weights.unsqueeze(-1), dim=1)

        if task == "drug":
            return (self.task_experts[0](domain_output), expert_weights)
        elif task == "diag":
            return (self.task_experts[1](domain_output), expert_weights)
        else:
            task_outputs = [expert(domain_output) for expert in self.task_experts]
            return (task_outputs, expert_weights)

    def get_orthogonality_loss(self):

        all_weights = torch.stack([expert[0].weight for expert in self.domain_experts])

        flat_weights = all_weights.view(self.num_domains, -1)

        norm_weights = F.normalize(flat_weights, p=2, dim=1)

        cosine_matrix = torch.matmul(norm_weights, norm_weights.t())

        orth_loss = torch.sum(torch.abs(cosine_matrix)) - self.num_domains

        return orth_loss

