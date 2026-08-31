"""
Optuna hyperparameter search (Learning Rate + Weight Decay) for the
BASELINE Attention U-Net model, on the FINAL pipeline configuration:

    - Vahadane multi-target stain augmentation:  ON
    - Refined ("new") lymphocyte masks:          ON
    - Online geometric/intensity augmentations:  ON
    - Class-aware weighted sampler:               ON
    - Loss function:                              CE + Dice (CustomDiceCELoss)
    - max_epochs / batch_size:                    based on config 

"""

import os
import json
import datetime
import sys
import gc

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import optuna.visualization.matplotlib as optuna_viz

from torch.utils.data import WeightedRandomSampler, DataLoader

from monai.data import PersistentDataset, Dataset, decollate_batch
from monai.networks.nets import AttentionUnet
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotate90d, RandFlipd, RandAffined,
    RandGaussianNoised, AsDiscrete, Lambdad, RandGridDistortiond
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.custom_losses import CustomDiceCELoss
from utils.helpers import EarlyStopping, RandomStainStyleD

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def remap_classes(x):
    remapped = np.zeros_like(x, dtype=np.int64)
    remapped[x == 0] = 255
    remapped[x == 1] = 0
    remapped[x == 2] = 1
    remapped[x == 3] = 2
    remapped[x == 255] = 255
    return remapped


def scale_channels(base_channels, mult, divisor=8, min_ch=8):
    """Scale a base channel tuple by `mult`, rounding each stage to the
    nearest multiple of `divisor` (GPU-friendly), for coarse width search.
    E.g. (32,64,128,256,512) * 0.5 -> (16,32,64,128,256).
    """
    scaled = []
    for c in base_channels:
        c_new = int(round((c * mult) / divisor) * divisor)
        scaled.append(max(min_ch, c_new))
    return tuple(scaled)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_paths(df, output_dir, use_vahadane, use_new_masks):
    images, masks = [], []
    img_folder = 'images_aug' if use_vahadane else 'images'
    mask_folder = 'masks_train_betterlymphs' if use_new_masks else 'masks_train'
    for f in df['filename']:
        img_path = os.path.join(output_dir, img_folder, f)
        mask_path = os.path.join(output_dir, mask_folder, f)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)
    return images, masks


def prepare_data(cfg):
    OUTPUT_DIR = os.path.join(
        cfg.dataset.base_dir,
        f"overlap{cfg.dataset.overlap}_patchsize{cfg.dataset.patch_size}_fixed"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, "patches_metadata.json"), "r") as jsonfile:
        data = json.load(jsonfile)

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.rename(columns={'index': 'filename'}, inplace=True)

    patch_area = cfg.dataset.patch_size * cfg.dataset.patch_size
    df['tissue_px'] = df['tumor_px'] + df['stroma_px'] + df['lymphocytes_px']
    df['bg_ratio'] = df['bg_px'] / patch_area
    df_an = df[(df['tissue_px'] > 0)]

    wsi_stats = df_an.groupby('wsi_name')[['tumor_px', 'stroma_px', 'lymphocytes_px']].sum().reset_index()
    wsi_stats['total_px'] = wsi_stats['tumor_px'] + wsi_stats['stroma_px'] + wsi_stats['lymphocytes_px']
    wsi_stats['tumor_prop'] = wsi_stats['tumor_px'] / wsi_stats['total_px']
    wsi_stats['stroma_prop'] = wsi_stats['stroma_px'] / wsi_stats['total_px']
    wsi_stats['lymph_prop'] = wsi_stats['lymphocytes_px'] / wsi_stats['total_px']

    lymph_threshold = wsi_stats['lymph_prop'].quantile(0.70)
    stroma_threshold = wsi_stats['stroma_prop'].quantile(0.60)

    def assign_pseudo_class(row):
        if row['lymph_prop'] >= lymph_threshold:
            return 'High_Lymph'
        elif row['stroma_prop'] >= stroma_threshold:
            return 'High_Stroma'
        else:
            return 'High_Tumor'

    wsi_stats['stratify_group'] = wsi_stats.apply(assign_pseudo_class, axis=1)

    train_wsi_df, temp_wsi_df = train_test_split(
        wsi_stats, test_size=0.3, stratify=wsi_stats['stratify_group'], random_state=40
    )
    val_wsi_df, test_wsi_df = train_test_split(
        temp_wsi_df, test_size=0.3, stratify=temp_wsi_df['stratify_group'], random_state=40
    )

    # Optional: shrink to a stratified subset of WSIs for a faster HPO search
    # Set cfg.optuna.wsi_subset_frac < 1.0 to use.
    subset_frac = float(getattr(cfg.optuna, "wsi_subset_frac", 1.0))
    if subset_frac < 1.0:
        train_wsi_df, _ = train_test_split(
            train_wsi_df, train_size=subset_frac,
            stratify=train_wsi_df['stratify_group'], random_state=40
        )
        val_wsi_df, _ = train_test_split(
            val_wsi_df, train_size=subset_frac,
            stratify=val_wsi_df['stratify_group'], random_state=40
        )

    train_wsis = train_wsi_df['wsi_name'].tolist()
    val_wsis = val_wsi_df['wsi_name'].tolist()
    print(f"[Optuna tuning] WSIs used -> Train={len(train_wsis)}, Val={len(val_wsis)}")

    train_df = df_an[df_an['wsi_name'].isin(train_wsis)]
    val_df = df_an[df_an['wsi_name'].isin(val_wsis)]

    train_images, train_segs = get_paths(train_df, OUTPUT_DIR, True, True)
    val_images, val_segs = get_paths(val_df, OUTPUT_DIR, True, True)

    train_transforms = Compose([
        RandomStainStyleD(keys=["image"]),
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
        Lambdad(keys=["label"], func=remap_classes),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(np.pi / 4, np.pi / 4), scale_range=(0.2, 0.2),
            mode=("bilinear", "nearest"), padding_mode="reflection"
        ),
        RandGridDistortiond(
            keys=["image", "label"], prob=0.5,
            distort_limit=(-0.05, 0.05),
            mode=("bilinear", "nearest"), padding_mode="reflection"
        ),
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.05)
    ])

    val_load_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=["label"], func=remap_classes),
        ScaleIntensityd(keys=["image"]),
    ])

    train_files = [{"image": i, "label": s} for i, s in zip(train_images, train_segs)]
    val_files = [{"image": i, "label": s} for i, s in zip(val_images, val_segs)]

    total_tumor = train_df["tumor_px"].sum()
    total_stroma = train_df["stroma_px"].sum()
    total_lymph = train_df["lymphocytes_px"].sum()
    total_px = total_tumor + total_stroma + total_lymph + 1e-8

    w_t = total_px / (total_tumor + 1e-8)
    w_s = total_px / (total_stroma + 1e-8)
    w_l = total_px / (total_lymph + 1e-8)

    patch_w = (
        train_df['tumor_px'] * w_t +
        train_df['stroma_px'] * w_s +
        train_df['lymphocytes_px'] * w_l
    )
    patch_w = torch.DoubleTensor(patch_w.values)

    # Use a smaller num_samples/epoch during HPO if cfg.optuna.num_sampler is set,
    hpo_num_sampler = getattr(cfg.optuna, "num_sampler", None) or cfg.training.num_sampler
    sampler = WeightedRandomSampler(
        weights=patch_w, num_samples=hpo_num_sampler, replacement=True
    )

    cache_suffix = f"_{cfg.dataset.patch_size}_optuna"
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = PersistentDataset(val_files, transform=val_load_transforms, cache_dir=f"./cache_val{cache_suffix}")
    val_ds = Dataset(data=val_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=False,
        sampler=sampler, num_workers=cfg.training.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, OUTPUT_DIR



def train_and_evaluate(trial, cfg, lr, weight_decay, width_mult, train_loader, val_loader, device, num_classes):
    try:
        return _train_and_evaluate_inner(
            trial, cfg, lr, weight_decay, width_mult, train_loader, val_loader, device, num_classes
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[Trial {trial.number}] CUDA OOM at width_mult={width_mult} "
              f"(lr={lr:.2e}, wd={weight_decay:.2e}): {e}")
        trial.set_user_attr("failure_reason", "cuda_oom")
        raise optuna.TrialPruned()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[Trial {trial.number}] CUDA OOM (RuntimeError) at width_mult={width_mult}: {e}")
            trial.set_user_attr("failure_reason", "cuda_oom")
            raise optuna.TrialPruned()
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _train_and_evaluate_inner(trial, cfg, lr, weight_decay, width_mult, train_loader, val_loader, device, num_classes):
    base_channels = tuple(cfg.model.channels)
    channels = scale_channels(base_channels, width_mult)

    model = AttentionUnet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        channels=channels,
        strides=(2, 2, 2, 2),
        dropout=cfg.model.dropout
    ).to(device)

    n_params = count_params(model)
    trial.set_user_attr("width_mult", width_mult)
    trial.set_user_attr("channels", list(channels))
    trial.set_user_attr("num_params", n_params)
    print(f"[Trial {trial.number}] width_mult={width_mult} channels={channels} "
          f"params={n_params / 1e6:.2f}M")

    loss_function = CustomDiceCELoss(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.tmax)
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(
        patience=getattr(cfg.optuna, "patience", cfg.training.patience),
        mode='max', verbose=False
    )

    accum = cfg.training.accum
    best_val_dice = -1.0

    hpo_max_epochs = getattr(cfg.optuna, "max_epochs", None) or cfg.training.max_epochs

    for epoch in range(hpo_max_epochs):
        epoch_train_start = datetime.datetime.now()
        model.train()
        epoch_loss, step = 0, 0
        optimizer.zero_grad(set_to_none=True)
        train_pbar = tqdm(
            train_loader,
            desc=f"[Trial {trial.number}] lr={lr:.2e} wd={weight_decay:.2e} w={width_mult} "
                 f"Epoch {epoch + 1}/{hpo_max_epochs}",
            leave=False
        )

        for i, batch_data in enumerate(train_pbar):
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True).long().squeeze(1)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            loss = loss / accum

            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * accum
                step += 1
                train_pbar.set_postfix({"loss": f"{loss.item() * accum:.4f}"})

            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        scheduler.step()
        val_start = datetime.datetime.now()
        train_time_s = (val_start - epoch_train_start).total_seconds()
        val_dice_metric = DiceMetric(include_background=True, reduction="mean")

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                val_outputs = sliding_window_inference(
                    val_inputs, (cfg.dataset.patch_size, cfg.dataset.patch_size), 4,
                    model, overlap=cfg.dataset.overlap, mode="gaussian"
                )

                mask = (val_labels != 255).float()
                val_labels_clean = torch.where(val_labels == 255, torch.zeros_like(val_labels), val_labels)
                v_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)

                v_outputs_list = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(v_outputs)]
                v_labels_list = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(val_labels_clean)]
                mask_list = decollate_batch(mask)

                masked_v_outputs = [pred * m for pred, m in zip(v_outputs_list, mask_list)]
                masked_v_labels = [label * m for label, m in zip(v_labels_list, mask_list)]
                val_dice_metric(y_pred=masked_v_outputs, y=masked_v_labels)

        val_dice_val = val_dice_metric.aggregate().item()
        val_time_s = (datetime.datetime.now() - val_start).total_seconds()
        avg_train_loss = epoch_loss / step if step > 0 else 0
        print(f"[Trial {trial.number}] Epoch {epoch + 1}/{hpo_max_epochs} "
              f"| T.Loss: {avg_train_loss:.4f} | V.Dice: {val_dice_val:.4f} "
              f"| Train: {train_time_s:.1f}s | Val: {val_time_s:.1f}s")

        if val_dice_val > best_val_dice:
            best_val_dice = val_dice_val

        trial.report(val_dice_val, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        early_stopping(val_dice_val)
        if early_stopping.stop_training:
            break

    return best_val_dice


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_determinism(seed=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3

    train_loader, val_loader, OUTPUT_DIR = prepare_data(cfg)

    n_trials = cfg.optuna.n_trials
    lr_low, lr_high = cfg.optuna.lr_range
    wd_low, wd_high = cfg.optuna.wd_range
    study_name = cfg.optuna.study_name
    storage = f"sqlite:///{os.path.join(OUTPUT_DIR, study_name)}.db"

    hpo_max_epochs = getattr(cfg.optuna, "max_epochs", None) or cfg.training.max_epochs
    pruner = HyperbandPruner(
        min_resource=cfg.optuna.hyperband_min_resource,
        max_resource=hpo_max_epochs,
        reduction_factor=cfg.optuna.hyperband_reduction_factor
    )
    sampler = TPESampler(seed=40)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    width_choices = list(cfg.optuna.width_mult_choices)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", lr_low, lr_high, log=True)
        weight_decay = trial.suggest_float("weight_decay", wd_low, wd_high, log=True)
        width_mult = trial.suggest_categorical("width_mult", width_choices)
        return train_and_evaluate(
            trial, cfg, lr, weight_decay, width_mult, train_loader, val_loader, device, num_classes
        )

    study.optimize(objective, n_trials=n_trials, timeout=cfg.optuna.timeout_seconds)

    print("Best trial:")
    print(f"Value (Val Mean Dice): {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Trials: {len(study.trials)} total | {len(completed)} completed | {len(pruned)} pruned")

    result = {
        "study_name": study_name,
        "timestamp": str(datetime.datetime.now()),
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "lr_range": [lr_low, lr_high],
        "wd_range": [wd_low, wd_high],
    }
    with open(os.path.join(OUTPUT_DIR, f"{study_name}_best_params.json"), "w") as f:
        json.dump(result, f, indent=4)

    try:
        ax = optuna_viz.plot_optimization_history(study)
        ax.figure.savefig(os.path.join(OUTPUT_DIR, f"{study_name}_optimization_history.png"))
        plt.close(ax.figure)

        ax2 = optuna_viz.plot_param_importances(study)
        ax2.figure.savefig(os.path.join(OUTPUT_DIR, f"{study_name}_param_importance.png"))
        plt.close(ax2.figure)
    except Exception as e:
        print(f"Could not render Optuna plots: {e}")

    try:
        rows = []
        for t in completed:
            n_params = t.user_attrs.get("num_params")
            width_mult = t.user_attrs.get("width_mult")
            if n_params is not None:
                rows.append({
                    "trial": t.number,
                    "val_dice": t.value,
                    "num_params": n_params,
                    "width_mult": width_mult,
                    "lr": t.params.get("lr"),
                    "weight_decay": t.params.get("weight_decay"),
                })

        if rows:
            eff_df = pd.DataFrame(rows).sort_values("num_params")
            eff_df.to_csv(os.path.join(OUTPUT_DIR, f"{study_name}_efficiency_trials.csv"), index=False)

            fig, ax3 = plt.subplots(figsize=(7, 5))
            scatter = ax3.scatter(
                eff_df["num_params"] / 1e6, eff_df["val_dice"],
                c=eff_df["width_mult"], cmap="viridis", s=50
            )
            ax3.set_xlabel("Parameters (M)")
            ax3.set_ylabel("Validation Mean Dice")
            ax3.set_title("Accuracy vs. model size across HPO trials")
            cbar = fig.colorbar(scatter, ax=ax3)
            cbar.set_label("width_mult")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f"{study_name}_accuracy_vs_size.png"))
            plt.close(fig)
    except Exception as e:
        print(f"Could not render accuracy-vs-size plot: {e}")


if __name__ == "__main__":
    main()