import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Task import initialize_task
from utils import seq_dataloader, set_random_seed, set_current_dataset
from my_baselines.my_baselines import *
from trainer import testing_multi_task, load_partial_state_dict
from preprocess.data_load import preprocess_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained ExpertCare checkpoint (F1 / Jaccard / PR-AUC / ROC-AUC).')
    parser.add_argument('--ckpt', type=str, default='./best_model/best_model.ckpt', help='path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--test_epochs', type=int, default=10, help='number of sampled test repetitions')
    args = parser.parse_args()

    set_current_dataset('mimic3')
    set_random_seed(args.seed)
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    data_args = argparse.Namespace(dataset='mimic3', developer=False, task='multi_task')
    drug_dataset = preprocess_data(data_args, task="drug_rec_ts")
    diag_dataset = preprocess_data(data_args, task="diag_pred_ts")

    Tv_drug, Tm_drug, tok_drug, size_drug = initialize_task(drug_dataset, argparse.Namespace(task="drug_rec_ts"))
    Tv_diag, Tm_diag, tok_diag, size_diag = initialize_task(diag_dataset, argparse.Namespace(task="diag_pred_ts"))

    _, _, test_drug = seq_dataloader(drug_dataset, batch_size=args.batch_size)
    _, _, test_diag = seq_dataloader(diag_dataset, batch_size=args.batch_size)

    model = Multi_DT(
        (Tv_drug, Tv_diag),
        (Tm_drug, Tm_diag),
        (size_drug, size_diag),
        device=device,
        embedding_dim=args.dim,
        dropout=args.dropout,
        task="multi_task",
        use_moe=True,
    )

    state = torch.load(args.ckpt, map_location=device)
    load_partial_state_dict(model, state)
    model = model.to(device)
    print(f'Loaded checkpoint: {args.ckpt}')

    out_drug, out_diag = testing_multi_task(
        test_drug, test_diag, args.test_epochs, model,
        (tok_drug, tok_diag), 0.8,
        {"drug_rec": "drugs", "diag_pred": "conditions"}, device)

    print('\n===== Drug Recommendation =====')
    print(out_drug)
    print('===== Diagnosis Prediction =====')
    print(out_diag)


if __name__ == '__main__':
    main()
