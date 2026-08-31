# WSI Efficient Tissue Segmentation

A deep-learning pipeline for segmenting tumor, stroma, and
lymphocyte regions in whole-slide cancer
histopathology images, built on the [TIGER challenge](https://tiger.grand-challenge.org/)
dataset. 

The pipeline covers artifact filtering and patch extraction, Vahadane stain
normalization, lymphocyte mask refinement, training, Optuna-based
hyperparameter search, and a four-way architecture comparison
(Attention U-Net baseline vs. ModernAttention, SegMamba, and U-KAN) on both
segmentation accuracy and computational cost.

## Results 

Ablation study on the baseline Attention U-Net (test-set Mean Dice, averaged
over Tumor / Stroma / Lymphocytes):

| Step                                   | Mean Dice |
|-----------------------------------------|:---------:|
| Unregularized baseline                  |  0.7155   |
| + weighted sampler + augmentations      |  0.7242   |
| + Vahadane stain augmentation           |  0.7471   |
| + refined lymphocyte masks              |  0.7584   |
| + Tversky loss (α=0.7, β=0.3), final    | **0.7614**|


Final four-architecture comparison:

| Model            | Mean Dice | Tumor Dice | Stroma Dice | Lymph Dice | Params (M) | ms/patch |
|------------------|:---------:|:----------:|:-----------:|:----------:|:----------:|:--------:|
| Attention U-Net (baseline) | 0.7554 | 0.8509 | 0.8267 | 0.5886 | 7.94  | 12.24 |
| ModernAttention  | **0.8055** | **0.8808** | **0.8737** | **0.6620** | 32.78 | 53.87 |
| SegMamba2        | 0.7496    | 0.8452 | 0.8405 | 0.5630 | **2.34** | 20.89 |
| U-KAN            | 0.7767    | 0.8639 | 0.8579 | 0.6084 | 6.36  | 14.52 |

**Note:** Unless otherwise specified (e.g., the final Tversky ablation step), all models were optimized using a custom combined Dice and Cross-Entropy (Dice+CE) loss function.

## Repository structure

```
.
├── configs/
│   └── config.yaml          # single Hydra config: dataset, training, model, ablation, optuna
├── requirements.txt
└── src/
    ├── main.py               # single CLI entry point
    ├── train.py               # single-run training 
    ├── tune_optuna.py         # Optuna LR / weight-decay / width search for the baseline
    ├── test.py                # final 4-architecture comparison: per-class pixel-wise Dice/Jaccard/P/R + params + ms/patch
    ├── eval.py                # per-run test-set evaluation used by train.py (Dice/Jaccard/P/R + overlays)
    ├── data/
    │   ├── patcher.py         # WSI -> patch extraction (HistoKit-style artifact filtering)
    │   ├── mask_repair.py     # lymphocyte mask refinement
    │   └── generate_augs.py   # offline Vahadane stain augmentation
    ├── models/
    │   ├── archs.py           # U-KAN
    │   ├── kan.py              # KANLinear / KAN base layers
    │   ├── utils_kan.py
    │   ├── mod_attention.py   # ModernAttention U-Net (GroupNorm, residual, LeakyReLU, Spatial2D Dropout)
    │   ├── seg_mamba2.py       # SegMamba2 upgraded segmamba arch. for future works
    │   └── legacy/
    │       └── seg_mamba.py    # SegMamba1 used version in this work.
    └── utils/
        ├── custom_losses.py    # CustomDiceCELoss, CustomTverskyCELoss
        └── helpers.py          # EarlyStopping, RandomStainStyleD
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Everything runs through `src/main.py`:

```bash
cd src

python main.py patch            # 1. WSI -> patches, artifact filtering, XML annotations -> masks_train/
python main.py repair-masks     # 2. lymphocyte mask refinement -> masks_train_betterlymphs/
python main.py augment          # 3. offline Vahadane stain augmentation -> images_aug/, masks_aug/
python main.py train            # 4. train one configuration (reads configs/config.yaml)
python main.py tune             # 5. Optuna HPO for the baseline (LR, weight decay, width)
python main.py test             # 6. final comparison across all four trained architectures
```

`train` and `tune` are Hydra jobs, so any config field can be overridden
straight from the command line without touching the YAML:

```bash
python main.py train model.type=UKAN ablation.loss_type=tversky ablation.tversky_alpha=0.7 ablation.tversky_beta=0.3
python main.py tune optuna.n_trials=100 optuna.timeout_seconds=259200 optuna.width_mult_choices=[0.5,0.75,1.0]
```

### Config layout (`configs/config.yaml`)

| Section     | Purpose |
|-------------|---------|
| `dataset`   | patch size, overlap, base directory of the extracted-patches dataset |
| `training`  | epochs, batch size, sampler size, scheduler `tmax`, early-stopping patience |
| `model`     | hyperparameters for **all four** architectures at once, change if needed (in this case it wasn't)|
| `ablation`  | ablation switches: `use_vahadane`, `use_new_masks`, `use_augs`, `use_sampler`, loss type + Tversky α/β |
| `optuna`    | HPO search space and budget for `tune_optuna.py` |

## Ablation flag reference

Each row of the ablation table corresponds to a combination of `ablation.*`
flags, all trained with `python main.py train`:

| Step | `ablation.*` flags |
|---|---|
| 1. Baseline | `use_sampler=false use_augs=false use_vahadane=false use_new_masks=false loss_type=dice_ce` |
| 2. + Weighted sampler | `use_sampler=true`, rest as above |
| 3. + Online augmentations | `use_augs=true`, rest as above |
| 4. + Vahadane (old masks) | `use_vahadane=true use_new_masks=false` |
| 5. + Vahadane (new masks) | `use_vahadane=true use_new_masks=true` |
| 6. + Tversky loss | `loss_type=tversky ablation.tversky_alpha=0.7 ablation.tversky_beta=0.3` |


## Open items

- **Dataset hardcoding:** The data parsing, extraction, and loading logic is currently hardcoded for the TIGER dataset. This design choice was necessitated by the highly specific nature of the problem—including custom nested XML annotations and specialized artifact filtering. Adapting the pipeline to other histopathology datasets will require refactoring the data ingestion modules.