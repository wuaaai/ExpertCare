import numpy as np
import torch
import torch.nn.functional as F
import traceback
from pyhealth.metrics import binary_metrics_fn
from tqdm import tqdm
from itertools import zip_longest

from utils import prepare_labels, get_sample_loader, log_outmemory, AsymmetricLossOptimized

def training_multi_task(data_loader_drug, data_loader_diag, model, label_tokenizer, optimizer, 
                        label_name_dict, log_outmemory_txt_path, device, loss_weights=(0.5, 0.5)):
    model.train()
    train_loss = 0
    batch_idx = 0
    asl_drug = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0).to(device)
    asl_diag = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1).to(device)
    with tqdm(total=max(len(data_loader_drug), len(data_loader_diag)), desc="Training multi-task", unit="batch") as pbar:
        for data_drug, data_diag in zip_longest(data_loader_drug, data_loader_diag):
            optimizer.zero_grad()
            try:
                current_loss = 0.0
                has_loss = False

                if data_drug is not None:
                    if isinstance(data_drug, dict):
                        label_drug = prepare_labels(data_drug[label_name_dict['drug_rec']], label_tokenizer[0]).to(device)
                    else:
                        label_drug = prepare_labels(data_drug[0][label_name_dict['drug_rec']], label_tokenizer[0]).to(device)

                    out_drug, _ = model(data_drug, task="drug")  
                    loss_drug = asl_drug(out_drug, label_drug)
                    current_loss += loss_weights[0] * loss_drug
                    has_loss = True

                if data_diag is not None:
                    if isinstance(data_diag, dict):
                        label_diag = prepare_labels(data_diag[label_name_dict['diag_pred']], label_tokenizer[1]).to(device)
                    else:
                        label_diag = prepare_labels(data_diag[0][label_name_dict['diag_pred']], label_tokenizer[1]).to(device)

                    out_diag, _ = model(data_diag, task="diag")  
                    loss_diag = asl_diag(out_diag, label_diag)
                    current_loss += loss_weights[1] * loss_diag
                    has_loss = True

                if has_loss and hasattr(model, "get_orthogonality_loss"):
                    orth_loss = model.get_orthogonality_loss()
                    current_loss += 0.01 * orth_loss

                if has_loss:
                    current_loss.backward()
                    optimizer.step()
                    train_loss += current_loss.item()
                    batch_idx += 1
                    avg_loss = train_loss / batch_idx
                else:
                    avg_loss = train_loss / (batch_idx + 1e-8)

            except torch.cuda.OutOfMemoryError:
                error_log = traceback.format_exc()
                log_outmemory({"drug_batch": data_drug, "diag_batch": data_diag}, error_log, log_outmemory_txt_path)
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                del data_drug, data_diag
                torch.cuda.empty_cache()
                continue

            if data_drug is not None:
                del data_drug, out_drug, label_drug, loss_drug
            if data_diag is not None:
                del data_diag, out_diag, label_diag, loss_diag
            if has_loss:
                del current_loss
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return avg_loss

def evaluating_multi_task(data_loader_drug, data_loader_diag, model, label_tokenizer, label_name_dict, device):

    model.eval()
    val_loss_drug, val_loss_diag = 0, 0
    y_t_all_drug, y_p_all_drug = [], []
    y_t_all_diag, y_p_all_diag = [], []

    with torch.no_grad():

        with tqdm(total=min(len(data_loader_drug), len(data_loader_diag)),
                  desc="Evaluating multi-task", unit="batch") as pbar:
            for batch_idx, (data_drug, data_diag) in enumerate(zip(data_loader_drug, data_loader_diag)):

                if isinstance(data_drug, dict):
                    label_drug = prepare_labels(data_drug[label_name_dict['drug_rec']], label_tokenizer[0]).to(device)
                else:
                    label_drug = prepare_labels(data_drug[0][label_name_dict['drug_rec']], label_tokenizer[0]).to(device)

                if isinstance(data_diag, dict):
                    label_diag = prepare_labels(data_diag[label_name_dict['diag_pred']], label_tokenizer[1]).to(device)
                else:
                    label_diag = prepare_labels(data_diag[0][label_name_dict['diag_pred']], label_tokenizer[1]).to(device)

                out_drug, _ = model(data_drug, task="drug")
                out_diag, _ = model(data_diag, task="diag")

                loss_drug = F.binary_cross_entropy_with_logits(out_drug, label_drug)
                loss_diag = F.binary_cross_entropy_with_logits(out_diag, label_diag)

                val_loss_drug += loss_drug.item()
                val_loss_diag += loss_diag.item()

                avg_loss_drug = val_loss_drug / (batch_idx + 1)
                avg_loss_diag = val_loss_diag / (batch_idx + 1)

                y_t_all_drug.append(label_drug.cpu().numpy())
                y_p_all_drug.append(torch.sigmoid(out_drug).cpu().numpy())

                y_t_all_diag.append(label_diag.cpu().numpy())
                y_p_all_diag.append(torch.sigmoid(out_diag).cpu().numpy())

                del data_drug, data_diag, out_drug, out_diag, loss_drug, loss_diag
                torch.cuda.empty_cache()

                pbar.update(1)
                pbar.set_postfix(avg_loss_drug=f"{avg_loss_drug:.4f}", avg_loss_diag=f"{avg_loss_diag:.4f}")

    y_true_drug = np.concatenate(y_t_all_drug, axis=0)
    y_prob_drug = np.concatenate(y_p_all_drug, axis=0)

    y_true_diag = np.concatenate(y_t_all_diag, axis=0)
    y_prob_diag = np.concatenate(y_p_all_diag, axis=0)

    metrics_drug = binary_metrics_fn(y_true_drug.ravel(), y_prob_drug.ravel(),
                                     metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    metrics_diag = binary_metrics_fn(y_true_diag.ravel(), y_prob_diag.ravel(),
                                     metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    val_loss_dict = {
        "drug_rec": avg_loss_drug,
        "diag_pred": avg_loss_diag
    }
    metrics_dict = {
        "drug_rec": metrics_drug,
        "diag_pred": metrics_diag
    }

    return val_loss_dict, metrics_dict

def testing_multi_task(drug_loader, diag_loader, test_epochs, model, label_tokenizer, sample_size, label_name_dict, device):
    results_drug = []
    results_diag = []

    metric_list = ['F1', 'Jaccard', 'PR-AUC', 'ROC-AUC']

    for epoch in range(test_epochs):
        print(f'\nTesting Epoch {epoch + 1}/{test_epochs}')
        sample_drug_loader = get_sample_loader(drug_loader, sample_size)
        sample_diag_loader = get_sample_loader(diag_loader, sample_size)

        val_loss_dict, metrics_dict = evaluating_multi_task(sample_drug_loader, sample_diag_loader, model,
                                                            label_tokenizer, label_name_dict, device)

        print(f'[Drug Prediction] F1: {metrics_dict["drug_rec"]["f1"]:.4f}, '
              f'Jaccard: {metrics_dict["drug_rec"]["jaccard"]:.4f}, '
              f'PR-AUC: {metrics_dict["drug_rec"]["pr_auc"]:.4f}, '
              f'ROC-AUC: {metrics_dict["drug_rec"]["roc_auc"]:.4f}')

        print(f'[Diagnosis Prediction] F1: {metrics_dict["diag_pred"]["f1"]:.4f}, '
              f'Jaccard: {metrics_dict["diag_pred"]["jaccard"]:.4f}, '
              f'PR-AUC: {metrics_dict["diag_pred"]["pr_auc"]:.4f}, '
              f'ROC-AUC: {metrics_dict["diag_pred"]["roc_auc"]:.4f}')

        results_drug.append(
            [metrics_dict["drug_rec"]["f1"], metrics_dict["drug_rec"]["jaccard"],
             metrics_dict["drug_rec"]["pr_auc"], metrics_dict["drug_rec"]["roc_auc"]]
        )

        results_diag.append(
            [metrics_dict["diag_pred"]["f1"], metrics_dict["diag_pred"]["jaccard"],
             metrics_dict["diag_pred"]["pr_auc"], metrics_dict["diag_pred"]["roc_auc"]]
        )

    results_drug = np.array(results_drug)
    results_diag = np.array(results_diag)

    mean_drug, std_drug = results_drug.mean(axis=0), results_drug.std(axis=0)
    mean_diag, std_diag = results_diag.mean(axis=0), results_diag.std(axis=0)

    outstring_drug = '【Drug Prediction】\n' + ''.join([
        "{}:\t{:.4f} ± {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean_drug, std_drug))
    ])

    outstring_diag = '【Diagnosis Prediction】\n' + ''.join([
        "{}:\t{:.4f} ± {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean_diag, std_diag))
    ])

    return outstring_drug, outstring_diag

def load_partial_state_dict(model, partial_state_dict):
    model_state = model.state_dict()
    for k, v in partial_state_dict.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state)

