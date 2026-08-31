'''Lymphocyte mask refinement tightens sparse lymphocyte
point-annotations to the actual cell boundaries found via adaptive
thresholding + connected components, using an EDT-nearest-label fill for
the remaining background gaps. Produces masks_train_betterlymphs/.'''

import json
import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


BASE_DIR = "/mnt/Data/jwandas/Code/dataset_patches/overlap0.5_patchsize512_fixed/"
OLD_MASKS_DIR = os.path.join(BASE_DIR, "masks_train")
NEW_MASKS_DIR = os.path.join(BASE_DIR, "masks_train_betterlymphs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")


def main():
    os.makedirs(NEW_MASKS_DIR, exist_ok=True)

    with open(os.path.join(BASE_DIR, "patches_metadata.json"), "r") as f:
        info_dict = json.load(f)

    for filename, patchinfo in tqdm(info_dict.items(), desc="Przetwarzanie masek"):

        old_mask_path = os.path.join(OLD_MASKS_DIR, filename)
        new_mask_path = os.path.join(NEW_MASKS_DIR, filename)

        if not os.path.exists(old_mask_path):
            continue

        if patchinfo.get("lymphocytes_px", 0) > 0:
            mask = cv2.imread(old_mask_path, cv2.IMREAD_GRAYSCALE)
            image_path = os.path.join(IMAGES_DIR, filename)
            image = cv2.imread(image_path)

            if mask is None or image is None:
                continue

            lymph_mask = (mask == 3).astype(np.uint8) * 255

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12)  # find lymphs

            cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel)

            _, labels = cv2.connectedComponents(thresh)
            valid_ids = np.unique(labels[lymph_mask == 255])
            valid_ids = valid_ids[valid_ids != 0]

            refined_lymph = np.isin(labels, valid_ids).astype(np.uint8) * 255

            holes = (mask == 3)  # boolean mask of lymph annotations

            distances, nearest_valid_indices = distance_transform_edt(holes, return_indices=True)  # find nearest value based on euclidian distance

            mask[holes] = mask[tuple(nearest_valid_indices)][holes]  # fill lymphs with nearest mask value

            mask[refined_lymph == 255] = 3

            cv2.imwrite(new_mask_path, mask)

        else:
            if patchinfo.get("tumor_px", 0) > 0 or patchinfo.get("stroma_px", 0) > 0:  # save only annotated patches
                shutil.copy2(old_mask_path, new_mask_path)


if __name__ == "__main__":
    main()
