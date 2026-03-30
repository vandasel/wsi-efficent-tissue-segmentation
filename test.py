import os
import json
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import pysqlite3

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from monai.data import DataLoader, PersistentDataset, decollate_batch
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Lambdad, AsDiscrete
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.seg_mamba import SegMamba
from models.archs import UKAN

OUTPUT_DIR = '/mnt/Data/jwandas/Code/dataset_patches/overlap0.5_patchsize512'
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3

# AttentionUnet
# BEST_LR = 0.00013426930782781682
# BEST_WEIGHT_DECAY = 1.939567127450755e-05
BEST_CHANNELS = (32, 64, 128, 256, 512)

PATCH_SIZE = 512
#UKAN
# BEST_LR = 0.0001397994927520457
# BEST_WEIGHT_DECAY = 2.5643518043677267e-06 
# BEST_CHANNELS_UKAN = (128, 160, 256)

#Mamba
# BEST_LR = 0.00017468508278525242
# BEST_WEIGHT_DECAY = 4.1608500989671645e-06
# BEST_F  = [48, 96, 192, 384]
# BEST_H  = 384
# BEST_DEPTHS =  [2, 2, 2, 2]

tests_output_dir = os.path.join(OUTPUT_DIR, 'tests/att512')
os.makedirs(tests_output_dir, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "patches_metadata.json"), "r") as jsonfile: 
    data = json.load(jsonfile)

df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df.rename(columns={'index': 'filename'}, inplace=True)

df_an = df[(df['tumor_px'] > 0) | (df['stroma_px'] > 0) | (df['lymphocytes_px'] > 0)]

wsi_stats = df_an.groupby('wsi_name')[['tumor_px', 'stroma_px', 'lymphocytes_px']].sum()
wsi_stats['dominant_class'] = wsi_stats.idxmax(axis=1)

train_wsis, temp_wsis = train_test_split(
    wsi_stats.index, 
    test_size=0.2, 
    stratify=wsi_stats['dominant_class'], 
    random_state=42
)

val_wsis, test_wsis = train_test_split(
    temp_wsis, 
    test_size=0.5, 
    stratify=wsi_stats.loc[temp_wsis, 'dominant_class'], 
    random_state=42
)

test_df = df_an[df_an['wsi_name'].isin(test_wsis)]

def get_paths(df):
    images = [os.path.join(OUTPUT_DIR, 'images', f) for f in df['filename']]
    masks = [os.path.join(OUTPUT_DIR, 'masks_train', f) for f in df['filename']]
    return images, masks

test_images, test_segs = get_paths(test_df)

def keep_raw_labels(x):
    remapped = np.zeros_like(x, dtype=np.int64)
    remapped[x == 0] = 255
    remapped[x == 1] = 0 
    remapped[x == 2] = 1  
    remapped[x == 3] = 2  
    remapped[x == 255] = 255
    return remapped

test_files = [{"image": i, "label": s, "filename": os.path.basename(i)} for i, s in zip(test_images, test_segs)]

load_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=True),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
    Lambdad(keys=["label"], func=keep_raw_labels),
    ScaleIntensityd(keys=["image"]),
])

test_ds = PersistentDataset(test_files, transform=load_transforms, cache_dir="./cache_test")
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)


model = AttentionUnet(
    spatial_dims=2,
    in_channels=3,
    out_channels=num_classes,
    channels=BEST_CHANNELS,
    strides=(2, 2, 2, 2)
).to(device)

# model = SegMamba(
#     in_chans=3,
#     out_chans=num_classes,  
#     spatial_dims=2,
#     feat_size = BEST_F,
#     hidden_size= BEST_H,
#     depths = BEST_DEPTHS
# ).to(device)

# model = UKAN(
#     num_classes=num_classes, 
#     input_channels=3,        
#     img_size=512,
#     embed_dims=BEST_CHANNELS_UKAN            
# ).to(device)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{model._get_name()}_{PATCH_SIZE}.pth"), map_location=device, weights_only=True))
model.eval()

dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

cmap = mcolors.ListedColormap(['red', 'blue', 'green', 'white']) 
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

with torch.no_grad():
    for test_data in tqdm(test_loader, desc="Testing"):
        inputs = test_data["image"].to(device)
        labels = test_data["label"].to(device)
        filename = test_data["filename"][0]

        outputs = sliding_window_inference(inputs, (256, 256), 4, model)
        
        mask = (labels != 255).float()
        labels_clean = torch.where(labels == 255, torch.zeros_like(labels), labels)
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        
        p_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(preds)]
        l_list = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(labels_clean)]
        mask_list = decollate_batch(mask)

        masked_p_list = [p * m for p, m in zip(p_list, mask_list)]
        masked_l_list = [l * m for l, m in zip(l_list, mask_list)]
        
        dice_metric_batch(y_pred=masked_p_list, y=masked_l_list)
 
        img_np = inputs[0].cpu().numpy().transpose(1, 2, 0)
        original_mask_np = labels[0].cpu().numpy()[0]
        raw_pred_np = preds[0].cpu().numpy()[0] 

        viz_mask_np = np.where(original_mask_np == 255, 3, original_mask_np)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(viz_mask_np, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1].set_title("GT Mask")
        axes[1].axis('off')
        
    
        axes[2].imshow(raw_pred_np, cmap=cmap, norm=norm, interpolation='nearest')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        plt.savefig(os.path.join(tests_output_dir, f"test_{filename}"), bbox_inches='tight')
        plt.close()

results = dice_metric_batch.aggregate()
class_names = ["Tumor", "Stroma", "Lymphocytes"]

print("Dice scores")
for i, name in enumerate(class_names):
    print(f"{name:<15}: {results[i].item():.4f}")
print("-"*30)
print(f"Mean Test Dice : {results.mean().item():.4f}")
print("="*30)