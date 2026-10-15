import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn

from .MoE import *

class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, feature_embeddings):

        attention_scores = self.attention_net(feature_embeddings).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), feature_embeddings).squeeze(1)
        return context_vector, attention_weights

class SeqAttention(nn.Module):
    def __init__(self, input_dim):
        super(SeqAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1, bias=False)
        )

    def forward(self, gru_outputs):

        scores = self.attention_net(gru_outputs).squeeze(-1)                           
        weights = F.softmax(scores, dim=1)                           
        context = torch.bmm(weights.unsqueeze(1), gru_outputs).squeeze(1)                              
        return context, weights
class Multi_DT(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            embedding_dim,
            dropout,
            task,            
            use_moe
    ):
        super(Multi_DT, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.use_moe = use_moe
        self.feature_attention = FeatureAttention(embedding_dim)
        self.seq_attention = SeqAttention(self.embedding_dim)

        self.multi_task = (task == "multi_task")
        self.num_tasks = 2 if self.multi_task else 1
        num_domains = 19

        if self.multi_task:

            self.visit_event_token = Tokenizers_visit_event                         
            self.monitor_event_token = Tokenizers_monitor_event                         
            self.output_size = output_size                        
        else:

            self.visit_event_token = [Tokenizers_visit_event]                                     
            self.monitor_event_token = [Tokenizers_monitor_event]

            self.output_size = [output_size]

        self.active_expert_ids = [0, 1, 7, 2]

        self.embeddings = nn.ModuleDict()

        for t in range(self.num_tasks):
            for feature_key in self.visit_event_token[t].keys():
                tokenizer = self.visit_event_token[t][feature_key]

                self.embeddings[f"task{t}_{feature_key}"] = nn.Embedding(
                    tokenizer.get_vocabulary_size(),
                    self.embedding_dim,
                    padding_idx=tokenizer.get_padding_index(),
                )

        self.visit_gru = nn.ModuleDict()

        for t in range(self.num_tasks):
            for feature_key in self.visit_event_token[t].keys():

                self.visit_gru[f"task{t}_{feature_key}"] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

            for feature_key in ['weight', 'age']:
                self.visit_gru[f"task{t}_{feature_key}"] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.monitor_gru = nn.ModuleDict()

        self.fc_age = nn.Linear(1, self.embedding_dim)
        self.fc_weight = nn.Linear(1, self.embedding_dim)

        item_num = len(self.visit_event_token[0].keys()) + 2                               

        if self.use_moe:
            self.fc_patient = MoE(

                input_dim=self.embedding_dim,
                output_dims=self.output_size,                    
                num_domains=num_domains,
                num_tasks=self.num_tasks,
                hidden_dim=128
            )
        else:

            if self.multi_task:
                self.fc_patient = nn.ModuleList([
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(self.embedding_dim, self.output_size[i])
                    )
                    for i in range(self.num_tasks)
                ])
            else:
                self.fc_patient = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.embedding_dim, self.output_size[0])
                )

        self.feature_visit_event_keys = list(self.visit_event_token[0].keys())
        self.feature_monitor_event_keys = list(self.monitor_event_token[0].keys())

    def forward(self, batch_data, task=None):

        if self.multi_task:
            if task is None:
                raise ValueError("Multi-task mode requires 'task' argument in forward")

            task_map = {'drug': 0, 'diag': 1}
            if task not in task_map:
                raise ValueError(f"Unknown task '{task}' for multi-task model")
            task_idx = task_map[task]
            task_prefix = f"task{task_idx}"
        else:

            task_idx = 0
            task_prefix = f"task{task_idx}" if self.num_tasks == 1 else ""

        batch_size = len(batch_data['visit_id'])
        patient_emb_list = []

        visit_keys = list(self.visit_event_token[task_idx].keys())
        for feature_key in visit_keys:
            x = self.visit_event_token[task_idx][feature_key].batch_encode_3d(batch_data[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=self.device)

            embed_key = f"task{task_idx}_{feature_key}"
            x = self.dropout(self.embeddings[embed_key](x))

            x = torch.sum(x, dim=2)
            gru_key = embed_key
            output, hidden = self.visit_gru[gru_key](x)
            feature_embedding, _ = self.seq_attention(output)
            patient_emb_list.append(feature_embedding)

        for feature_key in ['weight', 'age']:
            x = batch_data[feature_key]
            max_length = max(len(sublist) for sublist in x)
            x = [[float(item) for item in sublist] + [0] * (max_length - len(sublist)) for sublist in x]
            x = torch.tensor(x, dtype=torch.float, device=self.device)

            num_patients, num_visits = x.shape
            x = x.view(-1, 1)
            mask = (x == 0)

            if feature_key == 'weight':
                x = self.dropout(self.fc_weight(x))
            elif feature_key == 'age':
                x = self.dropout(self.fc_age(x))

            x = x * (~mask)
            x = x.view(num_patients, num_visits, -1)

            gru_key = f"task{task_idx}_{feature_key}"
            output, hidden = self.visit_gru[gru_key](x)

            patient_emb_list.append(hidden.squeeze(dim=0))

        domain_expert_ids = [self.active_expert_ids] * len(batch_data['visit_id'])

        patient_embs_stacked = torch.stack(patient_emb_list, dim=1)

        patient_emb, feature_weights = self.feature_attention(patient_embs_stacked)

        if self.use_moe:

            outputs, expert_weights = self.fc_patient(patient_emb, domain_expert_ids, task=task)

            if self.multi_task:

                return outputs, expert_weights
            else:

                return outputs, expert_weights
        else:

            if self.multi_task:
                return self.fc_patient[task_idx](patient_emb), None
            else:
                return self.fc_patient(patient_emb), None

