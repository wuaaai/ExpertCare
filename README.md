# ExpertCare

## Requirements

- Python 3.11
- CUDA-enabled GPU
- PyTorch, PyHealth, numpy, pandas, scikit-learn, tqdm, matplotlib, dill, joblib

```bash
pip install torch pyhealth numpy pandas scikit-learn tqdm matplotlib dill joblib
```

## Reproduce Results

```bash
python evaluate.py --ckpt ./best_model/best_model.ckpt
```

Expected output (mean ± std over 10 sampled test repetitions):

| Task | F1 | Jaccard | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Medication Recommendation | 0.6653 ± 0.0024 | 0.4985 ± 0.0027 | 0.7524 ± 0.0020 | 0.9236 ± 0.0006 |
| Disease Prediction | 0.4420 ± 0.0004 | 0.2837 ± 0.0003 | 0.4695 ± 0.0011 | 0.9180 ± 0.0005 |

## Train

```bash
python main.py --dataset mimic3 --task multi_task \
    --epochs 50 --test_epochs 10 --lr 5e-4 --batch_size 16 \
    --dim 128 --dropout 0.6 --device_id 0
```

Checkpoints and logs are saved to `./logs/`.
