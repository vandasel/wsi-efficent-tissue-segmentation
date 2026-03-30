import os
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm
from openslide import OpenSlide
from wholeslidedata import WholeSlideAnnotation
from PIL import Image
import concurrent.futures
import json


WSI_PATH = '/mnt/Data/jwandas/Code/wslData/tiger/'
XML_PATH = '/mnt/Data/jwandas/Data/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/'
MAT_PATH = '/mnt/Data/jwandas/Code/hist_res/masks_grandqc/'
PATCH_SIZE = 512
OVERLAP = 0.5
OUTPUT_DIR = f'/mnt/Data/jwandas/Code/dataset_patches/overlap{OVERLAP}_patchsize{PATCH_SIZE}'


TARGET_MAG = 10.0

ALLOW_LIST = [1, 7]
COLOR_MAP= {
    "rest" : (255,255,0),
    "stroma" : (0,0,255),
    "tumor" : (255,0,0),
    "lymphocytes_plasma" : (0,255,0)
}

os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks_rgb'), exist_ok=True) 
os.makedirs(os.path.join(OUTPUT_DIR, 'masks_train'), exist_ok=True)

def get_annotation_mask_cv2(wsa, x_lvl0, y_lvl0, region_size_lvl0, final_size):
    mask = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    scale_factor = final_size / region_size_lvl0

    patch_x1 = x_lvl0 + region_size_lvl0
    patch_y1 = y_lvl0 + region_size_lvl0
    d = {}
    order = ["rest", "stroma", "lymphocytes_plasma", "tumor"]
    for el in order:
        annotations = wsa.annotations_per_label.get(el)

        if not annotations:
            continue
            
        for ann in annotations:
            try:
                coords = np.array(ann.coordinates)
            except Exception:
                continue

            if coords.shape[0] < 3: 
                continue

            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)

            if max_x < x_lvl0 or min_x > patch_x1: continue
            if max_y < y_lvl0 or min_y > patch_y1: continue

            coords = coords - np.array([x_lvl0, y_lvl0])
            coords = coords * scale_factor
            
            pts = np.round(coords).astype(np.int32)
            
            pts[:, 0] = np.clip(pts[:, 0], 0, final_size - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, final_size - 1)

            cv2.fillPoly(mask, [pts], COLOR_MAP.get(ann.label.name))
    return mask

def process_slide(wsi_name):
    labels = {
    'invasive tumor': 1, # tp do klas glownych
    'in-situ tumor': 1,
    'inflamed stroma': 2,
    'tumor-associated stroma': 2,
    'lymphocytes and plasma cells': 3,
    'rest': 0,
    'healthy glands': 0,
    'necrosis not in-situ': 0,
    'roi': 0
    }

    renamed_labels = {
        'tumor': 1,
        'stroma': 2,
        'lymphocytes_plasma': 3,
        'rest': 0
    }

    wsa = WholeSlideAnnotation(
        XML_PATH + wsi_name.replace(".tif",".xml"), 
        labels=labels, 
        renamed_labels=renamed_labels
    )
    wsi = OpenSlide(WSI_PATH+wsi_name)
    
    try:
        mat = sio.loadmat(MAT_PATH+wsi_name.replace(".tif",".mat"))
    except FileNotFoundError:
        print(f"Error: Could not find {MAT_PATH+wsi_name.replace('.tif','.mat')}")
        return f"Error: {wsi_name} not found", {}

    try:
        mpp_x = float(wsi.properties.get('openslide.mpp-x', 0.25))
        raw_mag = 10.0 / mpp_x
        mag_l0 = round(raw_mag / 5) * 5
    except:
        mag_l0 = 20.0 

    downsample_needed = mag_l0 / TARGET_MAG
    mat_scale = mat["scale_val"].flatten()[0] 
    bboxes = mat["bbox"]
    mask_arts = mat.get("mask_art", None)

    read_size_l0 = int(PATCH_SIZE * downsample_needed)
    step_l0 = int(read_size_l0 * (1 - OVERLAP))
    
    total_patches = 0
    slide_metadata = {} 

    for r in range(len(bboxes)):
        bbox_raw = bboxes[r]
        if isinstance(bbox_raw, np.void) or hasattr(bbox_raw, 'dtype'):
            bbox_raw = bbox_raw[0] if bbox_raw.ndim > 1 else bbox_raw
        
        bbox_l0 = (bbox_raw / mat_scale).astype(int)
        
        if bbox_l0.size == 4:
            x0, y0, x1, y1 = bbox_l0.flatten()
        else:
            continue

        if mask_arts is not None:
            if mask_arts.size == len(bboxes):
                region_mask_art = mask_arts.flat[r]
            else:
                region_mask_art = mask_arts
        else:
            region_mask_art = None

        for y in range(y0, y1 - step_l0 + 1, step_l0):
            for x in range(x0, x1 - step_l0 + 1, step_l0):
                try:
                    patch_img = wsi.read_region((x, y), 0, (read_size_l0, read_size_l0)).convert("RGB")
                except:
                    continue

                if read_size_l0 != PATCH_SIZE:
                    patch_img = patch_img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
                
                patch_arr = np.array(patch_img)

                if region_mask_art is not None:
                    offset_x_l0 = x - x0
                    offset_y_l0 = y - y0
                    
                    mask_x = int(offset_x_l0 * mat_scale)
                    mask_y = int(offset_y_l0 * mat_scale)
                    mask_w = int(read_size_l0 * mat_scale)
                    mask_h = int(read_size_l0 * mat_scale)
                    
                    sub_mask = region_mask_art[mask_y : mask_y+mask_h, mask_x : mask_x+mask_w]
                    
                    if sub_mask.size == 0:
                        continue

                    mask_allow = np.isin(sub_mask, ALLOW_LIST).astype(np.uint8)
                    mask_allow_resized = np.array(Image.fromarray(mask_allow).resize(
                        (PATCH_SIZE, PATCH_SIZE), Image.Resampling.NEAREST
                    ))

                    mask_rgb = np.repeat(mask_allow_resized[:, :, np.newaxis], 3, axis=2)
                    patch_arr = patch_arr * mask_rgb
                    patch_arr[np.all(patch_arr == [0, 0, 0], axis=-1)] = [255, 255, 255]

                    if np.mean(patch_arr) > 250:
                        continue

                mask_gt = get_annotation_mask_cv2(wsa, x, y, read_size_l0, PATCH_SIZE)
                mask_gt = mask_gt * mask_rgb

                mask_train = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
                mask_train[np.all(mask_gt == (255, 255, 0), axis=-1)] = 0
                mask_train[np.all(mask_gt == (255, 0, 0), axis=-1)] = 1     
                mask_train[np.all(mask_gt == (0, 0, 255), axis=-1)] = 2
                mask_train[np.all(mask_gt == (0, 255, 0), axis=-1)] = 3
                
     
                unique, counts = np.unique(mask_train, return_counts=True)
                pixel_counts = dict(zip(unique, counts))

                patch_stats = {
                    "wsi_name": wsi_name,
                    "x": x,
                    "y": y,
                    "rest_px": int(pixel_counts.get(0, 0)),
                    "tumor_px": int(pixel_counts.get(1, 0)),
                    "stroma_px": int(pixel_counts.get(2, 0)),
                    "lymphocytes_px": int(pixel_counts.get(3, 0)),
                    "bg_px": int(pixel_counts.get(255, 0))
                }

                fname = f"{wsi_name.replace('.tif', '')}_x{x}_y{y}.png"
                slide_metadata[fname] = patch_stats

                cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', fname), cv2.cvtColor(patch_arr, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'masks_rgb', fname), cv2.cvtColor(mask_gt, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'masks_train', fname), mask_train)
                
                total_patches += 1
   
    return f"Finished {wsi_name}: {total_patches} patches generated.", slide_metadata



def main():
    wsi_list = [f for f in os.listdir(WSI_PATH) if f.endswith('.tif')]
    print(f"Found {len(wsi_list)} slides to process.")
    max_workers = max(1, os.cpu_count() - 2) 
    print(f"Extraction with {max_workers} parallel workers")

    global_metadata = {} 

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_slide, wsi_name): wsi_name for wsi_name in wsi_list}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(wsi_list), desc="Processing WSIs"):
            try:
                result_msg, slide_metadata = future.result() 
                tqdm.write(result_msg)     

                if slide_metadata: 
                    global_metadata.update(slide_metadata)
                    
            except Exception as exc:
                wsi_name = futures[future]
                tqdm.write(f"Slide {wsi_name} generated an exception: {exc}")


    json_path = os.path.join(OUTPUT_DIR, f'patches_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(global_metadata, f, indent=4)
        
    print(f"\nZapisano metadane do {json_path}. N patchy: {len(global_metadata)}")

if __name__ == "__main__":
    main()