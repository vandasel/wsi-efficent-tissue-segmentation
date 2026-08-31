'''Final comparison of the four trained architectures (Attention U-Net
baseline, ModernAttention, SegMamba2, U-KAN) on the held-out test set:
pixel-wise Dice / Jaccard / Precision / Recall per class (Tumor, Stroma,
Lymphocytes), plus parameter count and per-patch inference time.'''

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Lambdad, ScaleIntensityd
from monai.inferers import sliding_window_inference

from models.mod_attention import ModernAttentionUNet
from models.legacy.seg_mamba import SegMamba
from models.archs import UKAN
from monai.networks.nets import AttentionUnet

OUTPUT_DIR = "/mnt/Data/jwandas/Code/dataset_patches/overlap0.5_patchsize512_fixed"

MODELS_WEIGHTS = {
    "AttentionUnet": "AttentionUnet_512_lr5e-05_dice_ce_S1_A1_V1_NM1_2026-08-06 20:39:17.822619.pth",
    "ModernAttention": "ModernAttentionUNet_512_lr5e-05_cedice_3200_2026-08-02 12:33:13.368950.pth",
    "SegMamba": "SegMamba_512_lr5e-05_tversky_S0_A1_V1_NM1_2026-08-13 17:55:10.613479.pth",
    "UKAN": "UKAN_512_lr5e-05_cedice_3200_2026-08-03 16:40:53.809015.pth",
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_model(name, device):
    if name == "AttentionUnet":
        return AttentionUnet(spatial_dims=2, in_channels=3, out_channels=3, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), dropout=0.3).to(device)
    elif name == "ModernAttention":
        return ModernAttentionUNet(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(device)
    elif name == "SegMamba":
        return SegMamba(in_chans=3, out_chans=3, spatial_dims=2, feat_size=[16, 32, 64, 128], hidden_size=128, depths=[2, 2, 2, 2], drop_path_rate=0.3).to(device)
    elif name == "UKAN":
        return UKAN(num_classes=3, input_channels=3, img_size=512, embed_dims=[128, 160, 256], depths=[2, 2, 2, 2], drop_rate=0.3, drop_path_rate=0.3).to(device)
    raise ValueError(f"Unknown model: {name}")


def remap_classes(x):
    remapped = np.zeros_like(x, dtype=np.int64)
    remapped[x == 0] = 255
    remapped[x == 1] = 0
    remapped[x == 2] = 1
    remapped[x == 3] = 2
    remapped[x == 255] = 255
    return remapped


def main():
    with open(os.path.join(OUTPUT_DIR, "patches_metadata.json"), "r") as jsonfile:
        data = json.load(jsonfile)

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.rename(columns={'index': 'filename'}, inplace=True)
    df_an = df[(df['tumor_px'] > 0) | (df['stroma_px'] > 0) | (df['lymphocytes_px'] > 0)]

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

    train_wsi_df, temp_wsi_df = train_test_split(wsi_stats, test_size=0.3, stratify=wsi_stats['stratify_group'], random_state=40)
    val_wsi_df, test_wsi_df = train_test_split(temp_wsi_df, test_size=0.3, stratify=temp_wsi_df['stratify_group'], random_state=40)

    test_wsis = test_wsi_df['wsi_name'].tolist()
    test_df = df_an[df_an['wsi_name'].isin(test_wsis)]

    test_images = [os.path.join(OUTPUT_DIR, 'images', f) for f in test_df['filename']]
    test_segs = [os.path.join(OUTPUT_DIR, 'masks_train_betterlymphs', f) for f in test_df['filename']]
    test_files = [{"image": i, "label": s, "filename": os.path.basename(i)} for i, s in zip(test_images, test_segs)]

    load_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
        Lambdad(keys=["label"], func=remap_classes),
        ScaleIntensityd(keys=["image"])
    ])

    test_base_ds = Dataset(data=test_files, transform=load_transforms)
    test_loader = DataLoader(test_base_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Tumor", "Stroma", "Lymphocytes"]
    results_summary = {}

    for model_name, weight_file in MODELS_WEIGHTS.items():
        weight_path = os.path.join(OUTPUT_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"Missing checkpoint for {model_name}: {weight_path}")
            continue

        model = get_model(model_name, device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()

        n_params = count_params(model)

        all_preds_flat = []
        all_labels_flat = []
        inference_times_ms = []

        with torch.no_grad():
            for test_data in tqdm(test_loader, desc=f"Testing {model_name}"):
                inputs = test_data["image"].to(device)
                labels = test_data["label"].to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                outputs = sliding_window_inference(inputs, (512, 512), 4, model, overlap=0.5, mode="gaussian")

                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_times_ms.append((time.perf_counter() - start) * 1000)

                preds = torch.argmax(outputs, dim=1, keepdim=True)
                mask = (labels != 255)

                preds_flat = preds[mask].cpu().numpy().flatten()
                labels_flat = labels[mask].cpu().numpy().flatten()
                all_preds_flat.append(preds_flat)
                all_labels_flat.append(labels_flat)

        y_true = np.concatenate(all_labels_flat)
        y_pred = np.concatenate(all_preds_flat)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        jaccard = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        mean_ms_per_patch = float(np.mean(inference_times_ms))

        print(f"\n{model_name}")
        for i, name in enumerate(class_names):
            print(f"{name:<15} | Dice: {dice[i]:.4f} | Jaccard: {jaccard[i]:.4f} | Prec: {precision[i]:.4f} | Recall: {recall[i]:.4f}")
        print(f"Mean Dice: {np.mean(dice):.4f} | Params: {n_params / 1e6:.2f}M | ms/patch: {mean_ms_per_patch:.2f}")

        results_summary[model_name] = {
            "Mean_Dice": float(np.mean(dice)),
            "Tumor_Dice": float(dice[0]),
            "Stroma_Dice": float(dice[1]),
            "Lymph_Dice": float(dice[2]),
            "Tumor_Jaccard": float(jaccard[0]),
            "Stroma_Jaccard": float(jaccard[1]),
            "Lymph_Jaccard": float(jaccard[2]),
            "Lymph_Precision": float(precision[2]),
            "Lymph_Recall": float(recall[2]),
            "Params_M": n_params / 1e6,
            "ms_per_patch": mean_ms_per_patch,
            "Confusion_Matrix": cm.tolist(),
        }

    with open(os.path.join(OUTPUT_DIR, "all_models_final_eval.json"), "w") as f:
        json.dump(results_summary, f, indent=4)


if __name__ == "__main__":
    main()
