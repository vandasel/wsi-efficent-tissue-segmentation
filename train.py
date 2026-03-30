import os
import json
import pandas as pd
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import hydra
from omegaconf import DictConfig, OmegaConf

import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.custom_losses import CustomDiceCELoss 
from utils.helpers import EarlyStopping
from models.seg_mamba import SegMamba
from models.archs import UKAN


def build_model(cfg, device, num_classes):
    model_type = cfg.model.type
    
    if model_type == "AttentionUnet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=num_classes,
            channels=tuple(cfg.model.channels),
            strides=(2, 2, 2, 2)
        ).to(device)
        
    elif model_type == "UKAN":
        return UKAN(
            num_classes=num_classes, 
            input_channels=3,        
            img_size=cfg.dataset.patch_size,
            embed_dims=list(cfg.model.embed_dims)            
        ).to(device)
        
    elif model_type == "SegMamba":
        return SegMamba(
            in_chans=3,
            out_chans=num_classes,  
            spatial_dims=2,
            feat_size=list(cfg.model.feat_size),
            hidden_size=cfg.model.hidden_size,
            depths=list(cfg.model.depths)
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

def get_paths(df, output_dir):
    images = [os.path.join(output_dir, 'images', f) for f in df['filename']]
    masks = [os.path.join(output_dir, 'masks_train', f) for f in df['filename']]
    return images, masks


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    
    OUTPUT_DIR = os.path.join(cfg.dataset.base_dir, f"overlap{cfg.dataset.overlap}_patchsize{cfg.dataset.patch_size}")
    TESTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'tests_eval')
    os.makedirs(TESTS_OUTPUT_DIR, exist_ok=True)
    
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

    model = build_model(cfg, device, num_classes)
    loss_function = CustomDiceCELoss(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    scaler = torch.amp.GradScaler('cuda')

    early_stopping = EarlyStopping(patience=cfg.training.patience, mode='max', verbose=True)

    best_metric = -1
    history_loss = []
    history_val_dice = []
    
    save_filename = f"{model._get_name()}_{cfg.dataset.patch_size}_lr{cfg.model.lr}.pth"
    save_path = os.path.join(OUTPUT_DIR, save_filename)

    for epoch in range(cfg.training.max_epochs):
        model.train()
        epoch_loss, step = 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.max_epochs} [Train]", leave=False)
        
        for batch_data in train_pbar:
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True).long().squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            if model._get_name() != "SegMamba":
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                step += 1
                display_loss = loss.item()
            else:
                display_loss = 0.0

            train_pbar.set_postfix({"loss": f"{display_loss:.4f}"})

        scheduler.step()
        model.eval()
        metric_calculator = DiceMetric(include_background=True, reduction="mean")
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                
                val_outputs = sliding_window_inference(
                    val_inputs, (cfg.dataset.patch_size, cfg.dataset.patch_size), 4, model, overlap=cfg.dataset.overlap
                )
                
                mask = (val_labels != 255).float()
                val_labels_clean = torch.where(val_labels == 255, torch.zeros_like(val_labels), val_labels)
                v_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                
                v_outputs_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(v_outputs)]
                v_labels_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(val_labels_clean)]
                mask_list = decollate_batch(mask)
                
                masked_v_outputs = [pred * m for pred, m in zip(v_outputs_list, mask_list)]
                masked_v_labels = [label * m for label, m in zip(v_labels_list, mask_list)]
                metric_calculator(y_pred=masked_v_outputs, y=masked_v_labels)

        metric_val = metric_calculator.aggregate().item()
        avg_loss = epoch_loss / step if step > 0 else 0
        print(f"Epoch {epoch+1}/{cfg.training.max_epochs} | Loss: {avg_loss:.4f} | Val Dice: {metric_val:.4f}")

        history_loss.append(avg_loss)
        history_val_dice.append(metric_val)

        early_stopping(metric_val)

        if metric_val > best_metric:
            best_metric = metric_val
            torch.save(model.state_dict(), save_path)
            
        if early_stopping.stop_training:
            break

    print(f"Best Val Dice: {best_metric:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history_loss) + 1), history_loss, marker='o', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history_val_dice) + 1), history_val_dice, marker='o', color='orange', label='Val Mean Dice')
    plt.title('Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model._get_name()}_{cfg.dataset.patch_size}_{cfg.model.lr}_training_curves.png"))
    plt.close()

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    dice_metric_batch.reset()

    class_names = ["Tumor", "Stroma", "Lymphocytes"]
    
    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Testing"):
            test_inputs = test_data["image"].to(device)  
            test_labels = test_data["label"].to(device)

            test_outputs = sliding_window_inference(test_inputs, (cfg.dataset.patch_size, cfg.dataset.patch_size), 4, model)
            
            mask = (test_labels != 255).float()
            labels_clean = torch.where(test_labels == 255, torch.zeros_like(test_labels), test_labels)
            preds = torch.argmax(test_outputs, dim=1, keepdim=True)
            
            p_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(preds)]
            l_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(labels_clean)]
            mask_list = decollate_batch(mask)

            masked_p_list = [p * m for p, m in zip(p_list, mask_list)]
            masked_l_list = [l * m for l, m in zip(l_list, mask_list)]
            dice_metric_batch(y_pred=masked_p_list, y=masked_l_list)

    results = dice_metric_batch.aggregate()
    
    test_metrics_dict = {}
    for i, name in enumerate(class_names):
        val = results[i].item()
        test_metrics_dict[name] = val
        print(f"{name:<15}: {val:.4f}")
        
    mean_test_dice = results.mean().item()
    test_metrics_dict["Mean_Test_Dice"] = mean_test_dice
    print(f"Mean Test Dice: {mean_test_dice:.4f}")

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "training_history": {
            "loss": history_loss,
            "val_dice": history_val_dice
        },
        "best_val_dice": best_metric,
        "test_results": test_metrics_dict
    }

    json_filename = f"{model._get_name()}_{cfg.dataset.patch_size}_lr{cfg.model.lr}_results.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
        

if __name__ == "__main__":
    main()