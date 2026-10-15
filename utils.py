import random
import numpy as np
import matplotlib
import argparse
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_to_multihot(label, num_labels: int) -> torch.tensor:
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def log_params(dataset, args, log_path):
    argsDict = args.__dict__
    with open(log_path, 'w') as log_file:
        log_file.write(f'parameter of dataset:\n')
        log_file.write(f'{dataset} \n')
        log_file.write('--------------------------------------\n')
        log_file.write(f'args params:\n')
        for eachArg, value in argsDict.items():
            log_file.writelines(eachArg + ' : ' + str(value) + '\n')
        log_file.write('--------------------------------------\n')

def log_results_multi_task(epoch, run_time, train_loss, val_loss_dict, metrics_dict, log_path):

    with open(log_path, 'a') as log_file:
        log_file.write(f'Epoch {epoch + 1}\n')
        log_file.write(f'Train Loss: {train_loss:.4f}\n')

        if isinstance(val_loss_dict, dict):
            val_loss_str = ', '.join([f'{task}: {loss:.4f}' for task, loss in val_loss_dict.items()])
        else:
            val_loss_str = f'{val_loss_dict:.4f}'

        log_file.write(f'Validation Loss: {val_loss_str}\n')

        for task_name, metrics in metrics_dict.items():
            log_file.write(f'=== Task: {task_name} ===\n')
            log_file.write(f'F1: {metrics["f1"]:.4f}, '
                           f'Jaccard: {metrics["jaccard"]:.4f}, '
                           f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                           f'PR-AUC: {metrics["pr_auc"]:.4f}\n')

        log_file.write(f'Run_Time: {run_time:.3f}\n')

        if (epoch + 1) % 20 == 0:
            log_file.write(f'{epoch + 1} epoch Model saved !!!\n')
        log_file.write('--------------------------------------\n')

def log_outmemory(data, error_log, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'{error_log}\n')
        log_file.write('--------------------------------------\n')

def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels

def get_init_tokenizers(task_dataset, keys=None):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers

def seq_dataloader(dataset, split_ratio=[0.75, 0.1, 0.15], batch_size=64):
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, split_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

def get_sample_loader(data_loader, sample_size):
    sample_size = round(len(data_loader.dataset) * sample_size)
    dataset = data_loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)          
    sample_indices = indices[:sample_size]                      
    subset = Subset(dataset, sample_indices)
    sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=True,
                               collate_fn=data_loader.collate_fn)
    return sample_loader

def plot_losses(epoch_list, train_losses, val_losses, png_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_losses, label='Train Loss')
    plt.plot(epoch_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    min_epoch = epoch_list[min_val_loss_idx]

    plt.scatter(min_epoch, min_val_loss, color='red', zorder=5)              
    plt.text(min_epoch, min_val_loss, f'Min Val Loss\n({min_epoch:.0f}, {min_val_loss:.4f})', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=9, color='red')

    plt.autoscale(True)

    plt.savefig(png_path)
    plt.close()

def print_dataset_parameters(task_datasets, Tokenizers_visit_events, Tokenizers_monitor_events, label_sizes, args):

    multi_task = (args.task == "multi_task")

    if multi_task:

        outputs = []
        for i, task_name in enumerate(['drug_rec', 'diag_pred']):
            task_dataset = task_datasets[i]
            Tokenizers_visit_event = Tokenizers_visit_events[i]
            Tokenizers_monitor_event = Tokenizers_monitor_events[i]
            label_size = label_sizes[i]

            patient_num = len(task_dataset.patient_to_index)
            visit_num = len(task_dataset.visit_to_index)

            if task_name == 'drug_rec':
                cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
                drug_num = label_size
            elif task_name == 'diag_pred':
                cond_num = label_size
                drug_num = len(Tokenizers_visit_event['drugs'].vocabulary)
            else:
                cond_num = drug_num = 0

            proc_num = len(Tokenizers_visit_event['procedures'].vocabulary)
            lab_num = len(Tokenizers_monitor_event['lab_item'].vocabulary)
            inj_num = len(Tokenizers_monitor_event['inj_item'].vocabulary)

            cond = proc = drug = 0
            for visit in task_dataset.samples:
                if task_name == 'drug_rec':
                    cond += len(visit['conditions'][-1])
                    proc += len(visit['procedures'][-1])
                    drug += len(visit['drugs'])
                elif task_name == 'diag_pred':
                    cond += len(visit['conditions'])
                    proc += len(visit['procedures'][-1])
                    drug += len(visit['drugs'][-1])

            avg_visit = visit_num / patient_num if patient_num > 0 else 0
            avg_cond = cond / visit_num if visit_num > 0 else 0
            avg_proc = proc / visit_num if visit_num > 0 else 0
            avg_drug = drug / visit_num if visit_num > 0 else 0

            output = (
                f"--- Task {i} ({task_name}) ---\n"
                f"patient_num: {patient_num}\n"
                f"visit_num: {visit_num}\n"
                f"cond_num: {cond_num}\n"
                f"drug_num: {drug_num}\n"
                f"proc_num: {proc_num}\n"
                f"lab_num: {lab_num}\n"
                f"inj_num: {inj_num}\n"
                f"avg_visit: {avg_visit}\n"
                f"avg_cond: {avg_cond}\n"
                f"avg_proc: {avg_proc}\n"
                f"avg_drug: {avg_drug}"
            )
            outputs.append(output)
        return "\n\n".join(outputs)

    else:

        task_dataset = task_datasets
        Tokenizers_visit_event = Tokenizers_visit_events
        Tokenizers_monitor_event = Tokenizers_monitor_events
        label_size = label_sizes

        patient_num = len(task_dataset.patient_to_index)
        visit_num = len(task_dataset.visit_to_index)

        if args.task == 'drug_rec':
            cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
            drug_num = label_size
        elif args.task == 'diag_pred' or args.task == 'diag_pred_1010':
            cond_num = label_size
            drug_num = len(Tokenizers_visit_event['drugs'].vocabulary)
        elif args.task == 'drug_rec_ts':
            cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
            drug_num = label_size
        else:
            cond_num = drug_num = 0

        proc_num = len(Tokenizers_visit_event['procedures'].vocabulary)
        lab_num = len(Tokenizers_monitor_event['lab_item'].vocabulary)
        inj_num = len(Tokenizers_monitor_event['inj_item'].vocabulary)

        cond = proc = drug = 0
        for visit in task_dataset.samples:
            if args.task == 'drug_rec':
                cond += len(visit['conditions'][-1])
                proc += len(visit['procedures'][-1])
                drug += len(visit['drugs'])
            elif args.task == 'diag_pred':
                cond += len(visit['conditions'])
                proc += len(visit['procedures'][-1])
                drug += len(visit['drugs'][-1])
            elif args.task == 'drug_rec_ts':
                cond += len(visit['conditions'][-1])
                proc += len(visit['procedures'][-1])
                drug += len(visit['drugs'])

        avg_visit = visit_num / patient_num if patient_num > 0 else 0
        avg_cond = cond / visit_num if visit_num > 0 else 0
        avg_proc = proc / visit_num if visit_num > 0 else 0
        avg_drug = drug / visit_num if visit_num > 0 else 0

        output = (
            f"patient_num: {patient_num}\n"
            f"visit_num: {visit_num}\n"
            f"cond_num: {cond_num}\n"
            f"drug_num: {drug_num}\n"
            f"proc_num: {proc_num}\n"
            f"lab_num: {lab_num}\n"
            f"inj_num: {inj_num}\n"
            f"avg_visit: {avg_visit}\n"
            f"avg_cond: {avg_cond}\n"
            f"avg_proc: {avg_proc}\n"
            f"avg_drug: {avg_drug}"
        )
        return output

class AsymmetricLossOptimized(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)                            
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos = one_sided_w * los_pos
            los_neg = one_sided_w * los_neg

        loss = - (los_pos + los_neg)
        return loss.mean()
_current_dataset_name = None
_is_set_by_main = False
def set_current_dataset(name: str):

    global _current_dataset_name, _is_set_by_main
    if name not in ['mimic3']:
        print(f"警告 (config.py): 设置了未知的数据集名称 '{name}'，请检查 main.py 的 choices。")
    _current_dataset_name = name
    _is_set_by_main = True         
    print(f"配置信息 (config.py): 数据集已由 main.py 设置为 -> {_current_dataset_name}")

def read_dataset_argument() -> str:

    global _current_dataset_name 

    if _is_set_by_main:

        return _current_dataset_name
    else:

        print("警告 (config.py - read_dataset_argument): main.py 尚未调用 set_current_dataset()。将尝试独立解析命令行参数 '--dataset'。")

        parser_temp = argparse.ArgumentParser(add_help=False)
        parser_temp.add_argument('--dataset',
                                 type=str,
                                 default="mimic3",
                                 choices=['mimic3'],
                                 help='临时解析器使用的帮助信息')

        try:

             cmd_args = sys.argv[1:] if len(sys.argv) > 1 else []
             args_known, _ = parser_temp.parse_known_args(cmd_args)

             _current_dataset_name = args_known.dataset
             print(f"配置信息 (config.py - read_dataset_argument): 独立解析结果 -> {_current_dataset_name}")
             return _current_dataset_name
        except Exception as e:
             print(f"错误 (config.py - read_dataset_argument): 独立解析命令行失败: {e}。返回默认值 'mimic3'。")
             _current_dataset_name = "mimic3"
             return _current_dataset_name

