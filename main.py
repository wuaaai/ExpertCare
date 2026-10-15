import argparse
from Task import initialize_task
from utils import *
from my_baselines.my_baselines import *
from trainer import training_multi_task, evaluating_multi_task, testing_multi_task, load_partial_state_dict
from preprocess.data_load import preprocess_data
from torch.optim.lr_scheduler import StepLR
import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=10, help='Number of epochs to test.')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')
parser.add_argument('--model', type=str, default="Multi_DT",
                    help='Core model: ExpertCare (class Multi_DT, module my_baselines/ExpertCare.py)')
parser.add_argument('--device_id', type=int, default=0, help="选gpu编号的")
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3'])
parser.add_argument('--task', type=str, default="multi_task", choices=['multi_task'])
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--dim', type=int, default=128, help='embedding dim')
parser.add_argument('--dropout', type=float, default=0.6, help='dropout rate')
parser.add_argument('--developer', type=bool, default=False, help='developer mode')
parser.add_argument('--test', type=bool, default=False, help='test mode')
parser.add_argument('--notes', type=str, default="ExpertCare_mimic3", help='notes')
parser.add_argument("--scheduler", type=bool, default=True, help="scheduler mode")
parser.add_argument("--gamma", type=float, default=0.2, help="scheduler parameter")
parser.add_argument("--step_size", type=int, default=37, help="step_size")

args = parser.parse_args()
set_current_dataset(args.dataset)

def main(args):
    if args.developer:
        args.epochs = 3
        args.test_epochs = 2
        args.batch_size = 2
    set_random_seed(args.seed)
    print('{}--{}--{}--{}'.format(args.model, args.task, args.dataset, args.batch_size))
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    drug_dataset = preprocess_data(args, task="drug_rec_ts")
    diag_dataset = preprocess_data(args, task="diag_pred_ts")

    Tokenizers_visit_event_drug, Tokenizers_monitor_event_drug, label_tokenizer_drug, label_size_drug = initialize_task(
        drug_dataset, argparse.Namespace(task="drug_rec"))
    Tokenizers_visit_event_diag, Tokenizers_monitor_event_diag, label_tokenizer_diag, label_size_diag = initialize_task(
        diag_dataset, argparse.Namespace(task="diag_pred"))

    Tokenizers_visit_event = (Tokenizers_visit_event_drug, Tokenizers_visit_event_diag)
    Tokenizers_monitor_event = (Tokenizers_monitor_event_drug, Tokenizers_monitor_event_diag)
    label_tokenizer = (label_tokenizer_drug, label_tokenizer_diag)
    label_size = (label_size_drug, label_size_diag)

    train_loader_drug, val_loader_drug, test_loader_drug = seq_dataloader(drug_dataset, batch_size=args.batch_size)
    train_loader_diag, val_loader_diag, test_loader_diag = seq_dataloader(diag_dataset, batch_size=args.batch_size)

    if args.model == 'Multi_DT':
        model = Multi_DT(
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            label_size,
            device=device,
            embedding_dim=args.dim,
            dropout=args.dropout,
            task=args.task,
            use_moe=True
        )
    else:
        print("模型不存在")
        return

    label_name = {'drug_rec': 'drugs', 'diag_pred': 'conditions'}

    dataset_output = print_dataset_parameters(
        (drug_dataset, diag_dataset),
        (Tokenizers_visit_event_drug, Tokenizers_visit_event_diag),
        (Tokenizers_monitor_event_drug, Tokenizers_monitor_event_diag),
        (label_size_drug, label_size_diag),
        args
    )
    print(dataset_output)

    folder_path = './logs/{}_{}_{}_{}_{}'.format(args.model, args.task, args.dataset, args.batch_size, args.notes)
    os.makedirs(folder_path, exist_ok=True)
    ckpt_path = f'{folder_path}/best_model.ckpt'
    png_path = f'{folder_path}/loss.png'
    txt_path = f'{folder_path}/final_result.txt'
    log_txt_path = f'{folder_path}/log.txt'
    log_outmemory_txt_path = f'{folder_path}/log_outmemory.txt'

    if not args.test:
        epoch_list = []
        train_losses = []
        val_losses = []

        log_params(dataset_output, args, log_txt_path)

        print('--------------------Begin Training--------------------')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if args.scheduler:
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        best = {"drug_rec": float('inf'), "diag_pred": float('inf')}
        best_model = {"drug_rec": None, "diag_pred": None}

        for epoch in range(args.epochs):
            start_time = time.time()
            print(f'\nTraining Epoch {epoch + 1}/{args.epochs}')
            model = model.to(device)

            train_loss = training_multi_task(
                train_loader_drug,
                train_loader_diag,
                model,
                label_tokenizer,
                optimizer,
                label_name,
                log_outmemory_txt_path,
                device
            )

            val_loss, metrics_dict = evaluating_multi_task(
                val_loader_drug,
                val_loader_diag,
                model,
                label_tokenizer,
                label_name,
                device
            )

            print("\n=== 药物推荐任务(drug_rec) ===")
            print(f'F1: {metrics_dict["drug_rec"]["f1"]:.4f}, '
                  f'Jaccard: {metrics_dict["drug_rec"]["jaccard"]:.4f}, '
                  f'PR-AUC: {metrics_dict["drug_rec"]["pr_auc"]:.4f}, '
                  f'ROC-AUC: {metrics_dict["drug_rec"]["roc_auc"]:.4f}')

            print("\n=== 疾病预测任务(diag_pred) ===")
            print(f'F1: {metrics_dict["diag_pred"]["f1"]:.4f}, '
                  f'Jaccard: {metrics_dict["diag_pred"]["jaccard"]:.4f}, '
                  f'PR-AUC: {metrics_dict["diag_pred"]["pr_auc"]:.4f}, '
                  f'ROC-AUC: {metrics_dict["diag_pred"]["roc_auc"]:.4f}')

            end_time = time.time()
            run_time = end_time - start_time

            log_results_multi_task(epoch, run_time, train_loss, val_loss, metrics_dict, log_txt_path)

            for task in ["drug_rec", "diag_pred"]:
                if val_loss[task] < best[task]:
                    best[task] = val_loss[task]
                    best_model[task] = model.state_dict()

            epoch_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(sum(val_loss.values()) / len(val_loss))

            plot_losses(epoch_list, train_losses, val_losses, png_path)

            if args.scheduler:
                scheduler.step()

            torch.save(best_model["drug_rec"], ckpt_path.replace(".ckpt", "_drug_rec.ckpt"))
            torch.save(best_model["diag_pred"], ckpt_path.replace(".ckpt", "_diag_pred.ckpt"))

        print('--------------------Begin Testing--------------------')

        sample_size = 0.8

        best_model_drug = torch.load(ckpt_path.replace(".ckpt", "_drug_rec.ckpt"))
        best_model_diag = torch.load(ckpt_path.replace(".ckpt", "_diag_pred.ckpt"))

        load_partial_state_dict(model, best_model_drug)
        model = model.to(device)
        outstring_drug = testing_multi_task(test_loader_drug, test_loader_diag, args.test_epochs,
                                            model, label_tokenizer, sample_size,
                                            {"drug_rec": "drugs", "diag_pred": "conditions"}, device)[0]

        load_partial_state_dict(model, best_model_diag)
        model = model.to(device)
        outstring_diag = testing_multi_task(test_loader_drug, test_loader_diag, args.test_epochs,
                                            model, label_tokenizer, sample_size,
                                            {"drug_rec": "drugs", "diag_pred": "conditions"}, device)[1]

        print("\nFinal test result (best_model drug):")
        print(outstring_drug)
        with open(txt_path.replace(".txt", "_drug_rec.txt"), 'w+') as f:
            f.write("model_path: " + ckpt_path.replace(".ckpt", "_drug_rec.ckpt") + '\n')
            f.write(outstring_drug)

        print("\nFinal test result (best_model diag):")
        print(outstring_diag)
        with open(txt_path.replace(".txt", "_diag_pred.txt"), 'w+') as f:
            f.write("model_path: " + ckpt_path.replace(".ckpt", "_diag_pred.ckpt") + '\n')
            f.write(outstring_diag)

if __name__ == '__main__':
    main(args)
