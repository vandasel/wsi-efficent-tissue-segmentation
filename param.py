import os
import json
import pandas as pd
import numpy as np
import torch
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import optuna
from monai.data import DataLoader, PersistentDataset, Dataset, decollate_batch
from monai.networks.nets import AttentionUnet
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotate90d, RandFlipd, RandAffined,
    RandAdjustContrastd, RandShiftIntensityd, 
    RandGaussianNoised, RandScaleIntensityd,  
    RandHistogramShiftd, RandGaussianSmoothd, 
    AsDiscrete, Lambdad
)
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

# from torch.utils.tensorboard import SummaryWriter

from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.custom_losses import CustomDiceCELoss 
from utils.helpers import EarlyStopping
from models.seg_mamba import SegMamba
from models.archs import UKAN

def remap_classes(x):
    remapped = np.zeros_like(x, dtype=np.int64)
    remapped[x == 0] = 255
    remapped[x == 1] = 0 
    remapped[x == 2] = 1  
    remapped[x == 3] = 2  
    remapped[x == 255] = 255
    return remapped

def get_paths(df, output_dir):
    images = [os.path.join(output_dir, 'images', f) for f in df['filename']]
    masks = [os.path.join(output_dir, 'masks_train', f) for f in df['filename']]
    return images, masks

def build_model(trial, cfg, device, num_classes):
    model_type = cfg.model.type
    
    if model_type == "SegMamba":
        channel_choice = trial.suggest_categorical("channels", ["small", "large"])
        if channel_choice == "small":
            f_sizes = [24, 48, 96, 192]
            h_size = 192  
        else:
            f_sizes = [48, 96, 192, 384]
            h_size = 384

        depth_choice = trial.suggest_categorical("depths", ["flat", "deep_bottleneck"])
        d = [2, 2, 2, 2] if depth_choice == "flat" else [2, 2, 4, 2] 

        model = SegMamba(
            in_chans=3, out_chans=num_classes, spatial_dims=2,
            feat_size=f_sizes, hidden_size=h_size, depths=d
        )

    elif model_type == "UKAN":
        channel_choice = trial.suggest_categorical("channels", ["small", "medium", "large"])
        if channel_choice == "small":
            embed_dims = [64, 128, 160]
        elif channel_choice == "medium":
            embed_dims = [128, 160, 256]
        else:
            embed_dims = [128, 256, 512]

        model = UKAN(
            num_classes=num_classes, input_channels=3,        
            img_size=cfg.dataset.patch_size, embed_dims=embed_dims            
        )

    elif model_type == "AttentionUnet":
        channel_choice = trial.suggest_categorical("channels", ["light", "standard"])
        if channel_choice == "light":
            channels = (16, 32, 64, 128, 256)
        else:
            channels = (32, 64, 128, 256, 512)

        model = AttentionUnet(
            spatial_dims=2, in_channels=3, out_channels=num_classes,
            channels=channels, strides=(2, 2, 2, 2)
        )
        
    else:
        raise ValueError(f"Bad model name: {model_type}")

    return model.to(device)

@hydra.main(version_base=None, config_path=".", config_name="config_tune")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    
    OUTPUT_DIR = os.path.join(cfg.dataset.base_dir, f"overlap{cfg.dataset.overlap}_patchsize{cfg.dataset.patch_size}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "patches_metadata.json"), "r") as jsonfile:
        data = json.load(jsonfile)

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.rename(columns={'index': 'filename'}, inplace=True)
    df_an = df[(df['tumor_px'] > 0) | (df['stroma_px'] > 0) | (df['lymphocytes_px'] > 0)]

    wsi_stats = df_an.groupby('wsi_name')[['tumor_px', 'stroma_px', 'lymphocytes_px']].sum()
    wsi_stats['dominant_class'] = wsi_stats.idxmax(axis=1)

    train_wsis, temp_wsis = train_test_split(wsi_stats.index, test_size=0.3, stratify=wsi_stats['dominant_class'], random_state=42)
    val_wsis, test_wsis = train_test_split(temp_wsis, test_size=0.4, stratify=wsi_stats.loc[temp_wsis, 'dominant_class'], random_state=42)

    train_df = df_an[df_an['wsi_name'].isin(train_wsis)]
    val_df = df_an[df_an['wsi_name'].isin(val_wsis)]
    test_df = df_an[df_an['wsi_name'].isin(test_wsis)]

    train_images, train_segs = get_paths(train_df, OUTPUT_DIR)
    val_images, val_segs = get_paths(val_df, OUTPUT_DIR)
    test_images, test_segs = get_paths(test_df, OUTPUT_DIR)

    load_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
        Lambdad(keys=["label"], func=remap_classes),
        ScaleIntensityd(keys=["image"]),
    ])

    train_random_transforms = Compose([
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandAffined(
            keys=["image", "label"], prob=0.3,
            rotate_range=(np.pi/18, np.pi/18), scale_range=(0.1, 0.1),            
            mode=("bilinear", "nearest"), padding_mode="reflection"          
        ),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.25)),
        RandHistogramShiftd(keys=["image"], prob=0.3, num_control_points=5),
        RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.15),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05)
    ])

    train_files = [{"image": i, "label": s} for i, s in zip(train_images, train_segs)]
    val_files = [{"image": i, "label": s} for i, s in zip(val_images, val_segs)]
    test_files = [{"image": i, "label": s, "filename": os.path.basename(i)} for i, s in zip(test_images, test_segs)]

    cache_suffix = f"_{cfg.dataset.patch_size}"
    train_base_ds = PersistentDataset(train_files, transform=load_transforms, cache_dir=f"./cache_train{cache_suffix}")
    val_ds = PersistentDataset(val_files, transform=load_transforms, cache_dir=f"./cache_val{cache_suffix}")
    test_ds = PersistentDataset(test_files, transform=load_transforms, cache_dir=f"./cache_test{cache_suffix}")

    train_ds = Dataset(data=train_base_ds, transform=train_random_transforms)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)

   
    def objective(trial):
        lr = trial.suggest_float("lr", cfg.tune.lr_min, cfg.tune.lr_max, log=True) 
        weight_decay = trial.suggest_float("weight_decay", cfg.tune.weight_decay_min, cfg.tune.weight_decay_max, log=True)    
        for param_name, param_value in trial.params.items():
            print(f" {param_name} : {param_value}")
        
        model = build_model(trial, cfg, device, num_classes)
        loss_function = CustomDiceCELoss(num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        scaler = torch.amp.GradScaler('cuda')

        best_metric = -1
        early_stopping = EarlyStopping(patience=cfg.training.patience, mode='max', verbose=False)


   

        for epoch in range(cfg.training.max_epochs):
            model.train()
            epoch_loss, step = 0, 0
            train_pbar = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{cfg.training.max_epochs}", leave=False)
            
            for batch_data in train_pbar:
                inputs = batch_data["image"].to(device, non_blocking=True)
                labels = batch_data["label"].to(device, non_blocking=True).long().squeeze(1)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)

                if not torch.isnan(loss):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    step += 1
            
            avg_loss = epoch_loss / step if step > 0 else 0


            model.eval()
            dice_metric.reset()

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    
                    val_outputs = model(val_inputs)
                    
                    mask = (val_labels != 255).float()
                    val_labels_clean = torch.where(val_labels == 255, torch.zeros_like(val_labels), val_labels)
                    v_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                    
                    v_outputs_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(v_outputs)]
                    val_labels_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(val_labels_clean)]
                    mask_list = decollate_batch(mask)
                    
                    masked_v_outputs = [pred * m for pred, m in zip(v_outputs_list, mask_list)]
                    masked_v_labels = [label * m for label, m in zip(val_labels_list, mask_list)]

                    dice_metric(y_pred=masked_v_outputs, y=masked_v_labels)

            metric = dice_metric.aggregate().item()
            print(f"Trial {trial.number} | Epoch {epoch+1}/{cfg.training.max_epochs} | Train Loss: {avg_loss:.4f} | Val Dice: {metric:.4f}")
     

            trial.report(metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            early_stopping(metric)

            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"optuna_best_{cfg.model.type}.pth"))

            if early_stopping.stop_training:
                break
      
        return best_metric

    pruner = optuna.pruners.HyperbandPruner(min_resource=6, max_resource=cfg.training.max_epochs, reduction_factor=3)
    study_name = f"tune_{cfg.model.type}_{cfg.dataset.patch_size}"
    storage_name = f"sqlite:///{os.path.join(OUTPUT_DIR, study_name)}.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="maximize", 
        pruner=pruner
    )
    study.optimize(
        objective, 
        callbacks=[MaxTrialsCallback(
            cfg.training.n_trials, 
            states=(TrialState.COMPLETE, TrialState.PRUNED)
        )]
    )
    print("Best params:", study.best_trial.params)
    print("Best val value:", study.best_value)

    best_results = {
        "model_type": cfg.model.type,
        "patch_size": cfg.dataset.patch_size,
        "best_val_dice": study.best_value,
        "best_params": study.best_trial.params
    }
    
    json_path = os.path.join(OUTPUT_DIR, f"optuna_results_{cfg.model.type}.json")
    with open(json_path, 'w') as f:
        json.dump(best_results, f, indent=4)
        

if __name__ == "__main__":
    main()