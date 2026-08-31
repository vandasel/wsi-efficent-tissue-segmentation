'''Single-configuration training run (Hydra job): WSI-level stratified
split, ablation-flag-driven data pipeline (Vahadane augmentation, mask
refinement, online augmentations, weighted sampler), model training with
early stopping, and final test-set evaluation via eval.py'''

import os
import json
import pandas as pd
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from eval import evaluate_model 
import hydra
from omegaconf import DictConfig, OmegaConf

import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from torch.utils.data import WeightedRandomSampler

from monai.data import PersistentDataset, Dataset, decollate_batch
from monai.networks.nets import AttentionUnet
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotate90d, RandFlipd, RandAffined,
    RandAdjustContrastd, RandShiftIntensityd, 
    RandGaussianNoised, RandScaleIntensityd,  
    RandHistogramShiftd, RandGaussianSmoothd, 
    AsDiscrete, Lambdad, RandRotated, RandZoomd, RandGridDistortiond
)
from monai.transforms import NormalizeIntensityd

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter
from monai.transforms import RandLambdad

from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.custom_losses import CustomDiceCELoss, CustomTverskyCELoss
from utils.helpers import EarlyStopping, RandomStainStyleD
from models.seg_mamba2 import SegMamba2
from models.archs import UKAN
from models.mod_attention import ModernAttentionUNet

import segmentation_models_pytorch as smp

import datetime
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def build_model(cfg, device, num_classes):
    model_type = cfg.model.type
    
    if model_type == "AttentionUnet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=num_classes,
            channels=tuple(cfg.model.channels),
            strides=(2, 2, 2, 2),
            dropout=cfg.model.dropout
        ).to(device)
        
    elif model_type == "UKAN":
        return UKAN(
            num_classes=num_classes, 
            input_channels=3,   
            img_size=512,      
            embed_dims=list(cfg.model.embed_dims),             
            depths=list(cfg.model.depths),        
            drop_rate=cfg.model.dropout,          
            drop_path_rate=cfg.model.drop_path_rate
        ).to(device)
        
    elif model_type == "SegMamba2":
        return SegMamba2(
            in_chans=3,
            out_chans=num_classes,  
            spatial_dims=2,
            feat_size=list(cfg.model.feat_size),
            hidden_size=cfg.model.hidden_size,
            depths=list(cfg.model.depths),
            drop_path_rate= cfg.model.drop_path_rate
        ).to(device)
    
    elif model_type == "ModAttention":
        return ModernAttentionUNet(
            in_channels=3,
            out_channels=num_classes,
            features=cfg.model.features
        ).to(device)

    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

def remap_classes(x):
    remapped = np.zeros_like(x, dtype=np.int64)
    remapped[x == 0] = 255
    remapped[x == 1] = 0 
    remapped[x == 2] = 1  
    remapped[x == 3] = 2  
    remapped[x == 255] = 255
    return remapped

def get_paths(df, output_dir, use_vahadane, use_new_masks):
    images = []
    masks = []
    
    img_folder = 'images_aug' if use_vahadane else 'images'
    
    mask_folder = 'masks_train_betterlymphs' if use_new_masks else 'masks_train'
        
    for f in df['filename']:
        img_path = os.path.join(output_dir, img_folder, f)
        mask_path = os.path.join(output_dir, mask_folder, f)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)
            
    return images, masks
   
def extract_transforms_to_dict(compose_obj):
    transforms_info = {}
    for i, t in enumerate(compose_obj.transforms): 
        name = type(t).__name__
        params = {k: str(v) for k, v in vars(t).items() if not k.startswith('_')}
        transforms_info[f"{i}_{name}"] = params
    return transforms_info

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    set_determinism(seed=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    
    OUTPUT_DIR = os.path.join(cfg.dataset.base_dir, f"overlap{cfg.dataset.overlap}_patchsize{cfg.dataset.patch_size}_fixed")
    TESTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'tests_eval')
    os.makedirs(TESTS_OUTPUT_DIR, exist_ok=True)
    
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

    print("\nWSI distribution before the split based on pseudo-class:")
    print(wsi_stats['stratify_group'].value_counts())

    train_wsi_df, temp_wsi_df = train_test_split(
        wsi_stats, 
        test_size=0.3, 
        stratify=wsi_stats['stratify_group'], 
        random_state=40
    )

    val_wsi_df, test_wsi_df = train_test_split(
        temp_wsi_df, 
        test_size=0.3, 
        stratify=temp_wsi_df['stratify_group'], 
        random_state=40
    )

    train_wsis = train_wsi_df['wsi_name'].tolist()
    val_wsis = val_wsi_df['wsi_name'].tolist()
    test_wsis = test_wsi_df['wsi_name'].tolist()

    print(f"WSI split: Train={len(train_wsis)}, Val={len(val_wsis)}, Test={len(test_wsis)}")

    train_df = df_an[df_an['wsi_name'].isin(train_wsis)]
    val_df = df_an[df_an['wsi_name'].isin(val_wsis)]
    test_df = df_an[df_an['wsi_name'].isin(test_wsis)]
    
    train_images, train_segs = get_paths(train_df, OUTPUT_DIR, cfg.ablation.use_vahadane, cfg.ablation.use_new_masks)
    val_images, val_segs = get_paths(val_df, OUTPUT_DIR, cfg.ablation.use_vahadane, cfg.ablation.use_new_masks)
    test_images, test_segs = get_paths(test_df, OUTPUT_DIR, cfg.ablation.use_vahadane, cfg.ablation.use_new_masks)

    if cfg.ablation.use_augs:
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
                rotate_range=(np.pi/4, np.pi/4), scale_range=(0.2, 0.2),            
                mode=("bilinear", "nearest"), padding_mode="reflection"          
            ),
            RandGridDistortiond(
                keys=["image", "label"], prob=0.5,
                distort_limit=(-0.05, 0.05),
                mode=("bilinear", "nearest"), padding_mode="reflection"
            ),
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.05)
        ])
    else:
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
            Lambdad(keys=["label"], func=remap_classes),
            ScaleIntensityd(keys=["image"])
        ])

    val_load_transforms = Compose([
        LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=["label"], func=remap_classes),
        ScaleIntensityd(keys=["image"]),
    ])

    augs_dict = extract_transforms_to_dict(train_transforms)

    train_files = [{"image": i, "label": s} for i, s in zip(train_images, train_segs)]
    val_files = [{"image": i, "label": s} for i, s in zip(val_images, val_segs)]
    test_files = [{"image": i, "label": s, "filename": os.path.basename(i)} for i, s in zip(test_images, test_segs)]

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
    print("Class distribution:")
    print(f"Tumor: {total_tumor/total_px:.3f}")
    print(f"Stroma: {total_stroma/total_px:.3f}")
    print(f"Lymphocytes: {total_lymph/total_px:.3f}")

    patch_w = torch.DoubleTensor(patch_w.values)

    if cfg.ablation.use_sampler:
        sampler = WeightedRandomSampler(weights=patch_w, num_samples=cfg.training.num_sampler, replacement=True)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True

    cache_suffix = f"_{cfg.dataset.patch_size}"
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = PersistentDataset(val_files, transform=val_load_transforms, cache_dir=f"./cache_val{cache_suffix}")
    test_ds = PersistentDataset(test_files, transform=val_load_transforms, cache_dir=f"./cache_test{cache_suffix}")

    val_ds = Dataset(data=val_ds)  
    test_ds = Dataset(data=test_ds)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=shuffle_flag, sampler=sampler, num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = build_model(cfg, device, num_classes)

    if cfg.ablation.loss_type == "tversky":
        loss_function = CustomTverskyCELoss(
            num_classes=num_classes, 
            alpha=cfg.ablation.tversky_alpha, 
            beta=cfg.ablation.tversky_beta
        ).to(device)
    else:
        loss_function = CustomDiceCELoss(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.tmax)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    scaler = torch.amp.GradScaler('cuda')

    early_stopping = EarlyStopping(patience=cfg.training.patience, mode='max', verbose=True)

    best_metric = -1
    history_train_loss = []
    history_val_loss = []
    history_train_dice = []
    history_val_dice = []
    
    exp_name = f"{model._get_name()}_{cfg.dataset.patch_size}_lr{cfg.model.lr}_{cfg.ablation.loss_type}_S{int(cfg.ablation.use_sampler)}_A{int(cfg.ablation.use_augs)}_V{int(cfg.ablation.use_vahadane)}_NM{int(cfg.ablation.use_new_masks)}"
    save_filename = f"{exp_name}_{str(datetime.datetime.now())}.pth"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    accum = cfg.training.accum 

    for epoch in range(cfg.training.max_epochs):
        model.train()
        epoch_loss, step = 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.max_epochs} [Train]", leave=False)
        
        train_dice_metric = DiceMetric(include_background=True, reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        
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
                display_loss = loss.item() * accum
            else:
                display_loss = 0.0

            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
    
            with torch.no_grad():
                mask_t = (labels != 255).float().unsqueeze(1)
                labels_clean_t = torch.where(labels == 255, torch.zeros_like(labels), labels).unsqueeze(1)
                preds_t = torch.argmax(outputs.detach(), dim=1, keepdim=True)
                
                p_list_t = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(preds_t)]
                l_list_t = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(labels_clean_t)]
                mask_list_t = decollate_batch(mask_t)
                
                masked_p_list_t = [p * m for p, m in zip(p_list_t, mask_list_t)]
                masked_l_list_t = [l * m for l, m in zip(l_list_t, mask_list_t)]
                train_dice_metric(y_pred=masked_p_list_t, y=masked_l_list_t)

            train_pbar.set_postfix({"loss": f"{display_loss:.4f}"})

        model.eval() 
        scheduler.step()
        val_dice_metric = DiceMetric(include_background=True, reduction="mean")
        val_loss_total, val_step = 0, 0
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                
                val_outputs = sliding_window_inference(
                    val_inputs, (cfg.dataset.patch_size, cfg.dataset.patch_size), 4, model, overlap=cfg.dataset.overlap, mode="gaussian"
                )
                
                v_loss = loss_function(val_outputs, val_labels.long().squeeze(1))
                if not torch.isnan(v_loss):
                    val_loss_total += v_loss.item()
                    val_step += 1
                
                mask = (val_labels != 255).float()
                val_labels_clean = torch.where(val_labels == 255, torch.zeros_like(val_labels), val_labels)
                v_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                
                v_outputs_list = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(v_outputs)]
                v_labels_list = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(val_labels_clean)]
                mask_list = decollate_batch(mask)
                
                masked_v_outputs = [pred * m for pred, m in zip(v_outputs_list, mask_list)]
                masked_v_labels = [label * m for label, m in zip(v_labels_list, mask_list)]
                val_dice_metric(y_pred=masked_v_outputs, y=masked_v_labels)

        avg_train_loss = epoch_loss / step if step > 0 else 0
        avg_val_loss = val_loss_total / val_step if val_step > 0 else 0
        
        train_dice_val = train_dice_metric.aggregate().item()
        val_dice_val = val_dice_metric.aggregate().item()
        
        print(f"Epoch {epoch+1}/{cfg.training.max_epochs} | T.Loss: {avg_train_loss:.4f} | V.Loss: {avg_val_loss:.4f} | T.Dice: {train_dice_val:.4f} | V.Dice: {val_dice_val:.4f}")
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_train_dice.append(train_dice_val)
        history_val_dice.append(val_dice_val)
     
        early_stopping(val_dice_val)
        
        if val_dice_val > best_metric:
            best_metric = val_dice_val
            torch.save(model.state_dict(), save_path)
            
        if early_stopping.stop_training:
            break

    print(f"Best Val Dice: {best_metric:.4f}")

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, marker='o', color='red', label='Val Loss')
    plt.title('Loss Curves (Check Overfitting)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history_train_dice) + 1), history_train_dice, marker='o', color='green', label='Train Mean Dice')
    plt.plot(range(1, len(history_val_dice) + 1), history_val_dice, marker='o', color='orange', label='Val Mean Dice')
    plt.title('Dice Score Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{exp_name}_training_curves_{str(datetime.datetime.now())}.png"))
    plt.close()
   
    test_metrics_dict = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        cfg=cfg,
        save_path=save_path,
        output_dir=OUTPUT_DIR,
        num_classes=num_classes
    )

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "online_augmentations": augs_dict, 
        "training_history": {
            "loss": history_train_loss,
            "val_dice": history_val_dice
        },
        "best_val_dice": best_metric,
        "test_results": test_metrics_dict
    }
  
    json_filename = f"{exp_name}_results_{str(datetime.datetime.now())}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
if __name__ == "__main__":
    main()