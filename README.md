# Bit by Bit - Software Track

```bash
bitbybit/
├── src/
│   └── bitbybit/
│       ├── kernel/          # hash‑kernel abstractions
│       │   ├── _base.py     # _HashKernel (provided)
│       │   ├── random.py    # RandomProjKernel (implement)
│       │   └── learned.py   # LearnedProjKernel (implement)
│       ├── nn/              # hash‑backed torch.nn layers (provided)
│       ├── utils/           # helpers (provided)
│       └── patch.py         # swaps torch.nn ⇄ bitbybit.nn
├── train.py                 # minimal training template
├── publish.py               # leaderboard uploader
├── config.py                # hashing layerwise config
└── requirements.txt
```

## Installation

```bash
uv sync -p 3.12
```

## Baseline Models

Weights are fetched from [https://github.com/chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) using the keys

```python
"cifar10_resnet20"
"cifar100_resnet20"
```

## Task Checklist

1. **Implement kernels**: complete `RandomProjKernel` and `LearnedProjKernel`.
2. **Train**: extend *train.py* to fine‑tune projections (and optionally the backbone).
3. **Submit**: save checkpoints to

   ```text
   software-track/submission_checkpoints/<model>.pth
   ```
   
   then run

   ```bash
   python publish.py \
     --team-name <team-name> \
     --key <pre-shared-key>  
   ```

Submissions are unlimited within the 24‑hour window; the server keeps your best score.

## Scoring

Evaluation lives in `bitbybit.utils.score.calculate_submission_score`.
