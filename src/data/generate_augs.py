'''Offline Vahadane multi-target stain augmentation, normalizes every patch against three reference templates
(eosinophilic-dominant "pink", basophilic-dominant "purple", canonical
"median"), writing the results plus copies of the originals to images_aug/
and masks_aug/, consumed at training time by RandomStainStyleD.'''

import os
import json
import pandas as pd
import shutil
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
from torch_staintools.normalizer import NormalizerBuilder


OUTPUT_DIR = "/mnt/Data/jwandas/Code/dataset_patches/overlap0.5_patchsize512_fixed/"

MASK_IN_PATH = os.path.join(OUTPUT_DIR, "masks_train_betterlymphs")
MASKS_OUT_PATH = os.path.join(OUTPUT_DIR, "masks_aug")

IMAGES_IN_PATH = os.path.join(OUTPUT_DIR, "images")
IMAGES_OUT_PATH = os.path.join(OUTPUT_DIR, "images_aug")

STYLES = ["median"]

TARGET_FILES = {
    "pink": "TCGA-EW-A1P1-01Z-00-DX1.4B670029-4B3B-4D76-8EA4-F4F29EEF9E37_x26376_y19709.png",
    "purple": "TCGA-BH-A18G-01Z-00-DX1.DB2B5819-CE83-4E07-BD03-2CD9CF2E246C_x8280_y1900.png",
    "median": "TC_S01_P000145_C0001_B102_x113650_y62744.png"
}


def main():
    os.makedirs(MASKS_OUT_PATH, exist_ok=True)
    os.makedirs(IMAGES_OUT_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "patches_metadata.json"), "r") as jsonfile:
        info_dict = json.load(jsonfile)

    df = pd.DataFrame.from_dict(info_dict, orient='index').reset_index()
    df.rename(columns={'index': 'filename'}, inplace=True)

    df_an = df[(df['tumor_px'] > 0) | (df['stroma_px'] > 0) | (df['lymphocytes_px'] > 0)]

    for filename in tqdm(df_an['filename'], desc="Oryginały"):
        try:
            shutil.copy2(os.path.join(MASK_IN_PATH, filename), os.path.join(MASKS_OUT_PATH, filename))
        except Exception:
            pass

    for style in STYLES:
        for filename in tqdm(df_an['filename'], desc=f"Mask generation: {style}"):
            mask_path_in = os.path.join(MASK_IN_PATH, filename)
            base_name = filename.replace(".png", "")
            mask_path_out = os.path.join(MASKS_OUT_PATH, f"{base_name}_{style}.png")

            try:
                shutil.copy2(mask_path_in, mask_path_out)
            except Exception:
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for filename in tqdm(df_an['filename'], desc="Oryginały (Obrazy)"):
        try:
            shutil.copy2(os.path.join(IMAGES_IN_PATH, filename), os.path.join(IMAGES_OUT_PATH, filename))
        except Exception:
            pass

    for style, target_name in TARGET_FILES.items():
        print(f"\n Processing with style: {style.upper()}")

        target_path = os.path.join(IMAGES_IN_PATH, target_name)
        if not os.path.exists(target_path):
            print(f"File doesnt exist: skipping")
            continue

        target_img = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
        target_tensor = ToTensor()(target_img).unsqueeze(0).to(device)

        normalizer = NormalizerBuilder.build('vahadane')
        normalizer.fit(target_tensor)

        for filename in tqdm(df_an['filename'], desc=f"Vahadane: {style}"):

            img_path = os.path.join(IMAGES_IN_PATH, filename)
            base_name = filename.replace(".png", "")
            out_path = os.path.join(IMAGES_OUT_PATH, f"{base_name}_{style}.png")

            try:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_tensor = ToTensor()(img).unsqueeze(0).to(device)

                norm_tensor = normalizer.transform(img_tensor)

                norm_img = ToPILImage()(norm_tensor.squeeze(0).cpu())
                norm_img_bgr = cv2.cvtColor(np.array(norm_img), cv2.COLOR_RGB2BGR)

                cv2.imwrite(out_path, norm_img_bgr)

            except Exception as e:
                pass


if __name__ == "__main__":
    main()
