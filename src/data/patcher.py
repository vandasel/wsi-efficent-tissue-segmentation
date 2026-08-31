'''WSI -> patch extraction: reads TIGER whole-slide images at a target
magnification, merges GrandQC artifact masks with XML tissue annotations
into a sliding-window patch grid, and writes each valid patch as an RGB
image plus an integer training mask (0=rest, 1=tumor, 2=stroma,
3=lymphocytes) to OUTPUT_DIR, alongside a patches_metadata.json summary.'''

import os
import json
import numpy as np
import scipy.io as sio
import cv2
from tqdm import tqdm
from openslide import OpenSlide
from wholeslidedata import WholeSlideAnnotation
from PIL import Image
import concurrent.futures


WSI_PATH = '/mnt/Data/jwandas/Code/wslData/tiger/'
XML_PATH = '/mnt/Data/jwandas/Data/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/'
MAT_PATH = '/mnt/Data/jwandas/Code/hist_res/masks_grandqc/'

PATCH_SIZE = 512
OVERLAP = 0.5
TARGET_MAG = 10.0

OUTPUT_DIR = f'/mnt/Data/jwandas/Code/dataset_patches/overlap{str(OVERLAP).replace(".", "p")}_patchsize{PATCH_SIZE}'

ALLOW_LIST = [1, 7]

COLOR_MAP = {
    "rest": (255, 255, 0),
    "stroma": (0, 0, 255),
    "tumor": (255, 0, 0),
    "lymphocytes_plasma": (0, 255, 0)
}

os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks_rgb'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks_train'), exist_ok=True)


LABELS = {
    'invasive tumor': 1,
    'in-situ tumor': 1,
    'inflamed stroma': 2,
    'tumor-associated stroma': 2,
    'lymphocytes and plasma cells': 3,
    'rest': 0,
    'healthy glands': 0,
    'necrosis not in-situ': 0,
    'roi': 0
}

RENAMED_LABELS = {
    'tumor': 1,
    'stroma': 2,
    'lymphocytes_plasma': 3,
    'rest': 0
}


def get_annotation_mask_cv2(wsa, x_lvl0, y_lvl0, region_size_lvl0, final_size):
    mask = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    scale_factor = final_size / region_size_lvl0
    patch_x1 = x_lvl0 + region_size_lvl0
    patch_y1 = y_lvl0 + region_size_lvl0

    order = ["rest", "stroma", "lymphocytes_plasma", "tumor"]

    for el in order:
        annotations = wsa.annotations_per_label.get(el)
        if not annotations:
            continue

        for ann in annotations:
            try:
                coords = np.asarray(ann.coordinates, dtype=np.float64)
            except Exception:
                continue

            if coords.ndim != 2 or coords.shape[0] < 3:
                continue

            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)

            if max_x < x_lvl0 or min_x > patch_x1:
                continue
            if max_y < y_lvl0 or min_y > patch_y1:
                continue

            coords = (coords - np.array([x_lvl0, y_lvl0])) * scale_factor
            pts = np.round(coords).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, final_size - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, final_size - 1)

            color = COLOR_MAP.get(ann.label.name)
            if color is not None:
                cv2.fillPoly(mask, [pts], color)

    return mask


def get_annotation_bbox(wsa):
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    found = False

    for label_name in ["rest", "stroma", "lymphocytes_plasma", "tumor"]:
        annotations = wsa.annotations_per_label.get(label_name, [])

        for ann in annotations:
            try:
                coords = np.asarray(ann.coordinates, dtype=np.float64)
            except Exception:
                continue

            if coords.ndim != 2 or coords.shape[0] < 3:
                continue

            found = True
            min_x = min(min_x, coords[:, 0].min())
            min_y = min(min_y, coords[:, 1].min())
            max_x = max(max_x, coords[:, 0].max())
            max_y = max(max_y, coords[:, 1].max())

    if not found:
        return None

    return int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x)), int(np.ceil(max_y))


def load_mat_data(mat_path):
    mat = sio.loadmat(mat_path)
    mat_scale = float(mat["scale_val"].flatten()[0])
    bboxes_raw = mat["bbox"]
    mask_arts = mat.get("mask_art", None)

    bboxes = []
    for r in range(len(bboxes_raw)):
        bbox_raw = bboxes_raw[r]

        if isinstance(bbox_raw, np.void) or hasattr(bbox_raw, 'dtype'):
            try:
                if bbox_raw.ndim > 1:
                    bbox_raw = bbox_raw[0]
            except Exception:
                pass

        bbox_raw = np.asarray(bbox_raw).flatten()
        if bbox_raw.size != 4:
            continue

        bbox_l0 = (bbox_raw / mat_scale).astype(int)
        x0, y0, x1, y1 = bbox_l0
        if x1 <= x0 or y1 <= y0:
            continue

        bboxes.append({"index": r, "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)})

    return bboxes, mask_arts, mat_scale


def get_region_mask_art(mask_arts, bbox_index, bbox_count):
    if mask_arts is None:
        return None
    try:
        if mask_arts.size == bbox_count:
            return mask_arts.flat[bbox_index]
    except Exception:
        pass
    return mask_arts


# Merges GrandQC-derived per-region artifact masks (mask_arts, keyed by MAT
# bounding box) into a single perpatch boolean mask, keeping only the
# tissue classes in ALLOW_LIST and resampling each region's mask into the
# patchs local coordinate frame.
def get_mat_mask_for_patch(x, y, read_size_l0, bboxes, mask_arts, mat_scale, final_size):
    mat_mask = np.zeros((final_size, final_size), dtype=np.uint8)
    if mask_arts is None:
        return mat_mask

    patch_x1 = x + read_size_l0
    patch_y1 = y + read_size_l0

    for bbox in bboxes:
        bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]

        if bx1 <= x or bx0 >= patch_x1:
            continue
        if by1 <= y or by0 >= patch_y1:
            continue

        region_mask_art = get_region_mask_art(mask_arts, bbox["index"], len(bboxes))
        if region_mask_art is None:
            continue

        region_h, region_w = region_mask_art.shape[:2]

        overlap_x0 = max(x, bx0)
        overlap_y0 = max(y, by0)
        overlap_x1 = min(patch_x1, bx1)
        overlap_y1 = min(patch_y1, by1)
        if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
            continue

        patch_x0 = int((overlap_x0 - x) * final_size / read_size_l0)
        patch_y0 = int((overlap_y0 - y) * final_size / read_size_l0)
        patch_x1_out = int((overlap_x1 - x) * final_size / read_size_l0)
        patch_y1_out = int((overlap_y1 - y) * final_size / read_size_l0)

        mask_x0 = int((overlap_x0 - bx0) * mat_scale)
        mask_y0 = int((overlap_y0 - by0) * mat_scale)
        mask_x1 = int((overlap_x1 - bx0) * mat_scale)
        mask_y1 = int((overlap_y1 - by0) * mat_scale)

        mask_x0 = max(0, min(mask_x0, region_w))
        mask_x1 = max(0, min(mask_x1, region_w))
        mask_y0 = max(0, min(mask_y0, region_h))
        mask_y1 = max(0, min(mask_y1, region_h))
        if mask_x1 <= mask_x0 or mask_y1 <= mask_y0:
            continue

        sub_mask = region_mask_art[mask_y0:mask_y1, mask_x0:mask_x1]
        if sub_mask.size == 0:
            continue

        allowed = np.isin(sub_mask, ALLOW_LIST).astype(np.uint8)
        target_w = patch_x1_out - patch_x0
        target_h = patch_y1_out - patch_y0
        if target_w <= 0 or target_h <= 0:
            continue

        allowed_resized = np.asarray(
            Image.fromarray(allowed).resize((target_w, target_h), Image.Resampling.NEAREST)
        )

        mat_mask[patch_y0:patch_y1_out, patch_x0:patch_x1_out] = np.maximum(
            mat_mask[patch_y0:patch_y1_out, patch_x0:patch_x1_out], allowed_resized
        )

    return mat_mask


def get_patch_grid(min_x, min_y, max_x, max_y, read_size_l0, step_l0):
    if max_x <= min_x or max_y <= min_y:
        return []

    xs, x = [], min_x
    while x < max_x:
        xs.append(int(x))
        x += step_l0

    ys, y = [], min_y
    while y < max_y:
        ys.append(int(y))
        y += step_l0

    return [(x, y) for y in ys for x in xs]


def process_slide(wsi_name):
    xml_path = os.path.join(XML_PATH, wsi_name.replace(".tif", ".xml"))
    mat_path = os.path.join(MAT_PATH, wsi_name.replace(".tif", ".mat"))

    if not os.path.exists(xml_path):
        return f"Error: XML not found for {wsi_name}", {}
    if not os.path.exists(mat_path):
        return f"Error: MAT not found for {wsi_name}", {}

    try:
        wsa = WholeSlideAnnotation(xml_path, labels=LABELS, renamed_labels=RENAMED_LABELS)
    except Exception as e:
        return f"Error loading XML {wsi_name}: {e}", {}

    try:
        wsi = OpenSlide(os.path.join(WSI_PATH, wsi_name))
    except Exception as e:
        return f"Error loading WSI {wsi_name}: {e}", {}

    try:
        bboxes, mask_arts, mat_scale = load_mat_data(mat_path)
    except Exception as e:
        return f"Error loading MAT {wsi_name}: {e}", {}

    if not bboxes:
        return f"Error: no valid MAT bboxes for {wsi_name}", {}

    try:
        mpp_x = float(wsi.properties.get('openslide.mpp-x', 0.25))
        raw_mag = 10.0 / mpp_x
        mag_l0 = round(raw_mag / 5) * 5
    except Exception:
        mag_l0 = 20.0

    # Steps across the tissue bounding box (MAT bboxes unioned with the XML
    # annotation bbox) with (1 - OVERLAP) stride.
    downsample_needed = mag_l0 / TARGET_MAG
    read_size_l0 = int(PATCH_SIZE * downsample_needed)
    step_l0 = max(1, int(read_size_l0 * (1 - OVERLAP)))

    mat_min_x = min(bbox["x0"] for bbox in bboxes)
    mat_min_y = min(bbox["y0"] for bbox in bboxes)
    mat_max_x = max(bbox["x1"] for bbox in bboxes)
    mat_max_y = max(bbox["y1"] for bbox in bboxes)

    annotation_bbox = get_annotation_bbox(wsa)
    if annotation_bbox is not None:
        ann_min_x, ann_min_y, ann_max_x, ann_max_y = annotation_bbox
        min_x = min(mat_min_x, ann_min_x)
        min_y = min(mat_min_y, ann_min_y)
        max_x = max(mat_max_x, ann_max_x)
        max_y = max(mat_max_y, ann_max_y)
    else:
        min_x, min_y, max_x, max_y = mat_min_x, mat_min_y, mat_max_x, mat_max_y

    patch_positions = get_patch_grid(min_x, min_y, max_x, max_y, read_size_l0, step_l0)

    total_patches = 0
    slide_metadata = {}

    for x, y in patch_positions:
        try:
            patch_img = wsi.read_region((x, y), 0, (read_size_l0, read_size_l0)).convert("RGB")
        except Exception:
            continue

        if patch_img.size != (read_size_l0, read_size_l0):
            continue

        if read_size_l0 != PATCH_SIZE:
            patch_img = patch_img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)

        patch_arr = np.asarray(patch_img)

        mat_mask = get_mat_mask_for_patch(x, y, read_size_l0, bboxes, mask_arts, mat_scale, PATCH_SIZE)
        mask_gt = get_annotation_mask_cv2(wsa, x, y, read_size_l0, PATCH_SIZE)
        annotation_pixels = np.any(mask_gt != 0, axis=2)
        mat_allowed = mat_mask > 0

        keep_pixels = mat_allowed.copy()
        keep_pixels[annotation_pixels] = True

        tissue_pixels = np.count_nonzero(keep_pixels)
        annotation_pixel_count = np.count_nonzero(annotation_pixels)

        if annotation_pixel_count == 0 and tissue_pixels == 0:
            continue

        patch_mask = keep_pixels.astype(np.uint8)
        patch_arr = patch_arr * patch_mask[:, :, np.newaxis]
        patch_arr[np.all(patch_arr == [0, 0, 0], axis=-1)] = [255, 255, 255]

        if annotation_pixel_count == 0 and np.mean(patch_arr) > 250:
            continue

        # RGB annotation mask -> single-channel integer training mask.
        # 255 = unannotated/ignore, 0 = rest, 1 = tumor, 2 = stroma, 3 = lymphocytes
        mask_train = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
        rest_mask = np.all(mask_gt == (255, 255, 0), axis=-1)
        tumor_mask = np.all(mask_gt == (255, 0, 0), axis=-1)
        stroma_mask = np.all(mask_gt == (0, 0, 255), axis=-1)
        lymphocyte_mask = np.all(mask_gt == (0, 255, 0), axis=-1)

        mask_train[rest_mask] = 0
        mask_train[tumor_mask] = 1
        mask_train[stroma_mask] = 2
        mask_train[lymphocyte_mask] = 3

        unique, counts = np.unique(mask_train, return_counts=True)
        pixel_counts = dict(zip(unique, counts))

        fname = f"{wsi_name.replace('.tif', '')}_x{x}_y{y}.png"

        slide_metadata[fname] = {
            "wsi_name": wsi_name,
            "x": int(x),
            "y": int(y),
            "rest_px": int(pixel_counts.get(0, 0)),
            "tumor_px": int(pixel_counts.get(1, 0)),
            "stroma_px": int(pixel_counts.get(2, 0)),
            "lymphocytes_px": int(pixel_counts.get(3, 0)),
            "bg_px": int(pixel_counts.get(255, 0)),
            "mat_allowed_px": int(np.count_nonzero(mat_allowed)),
            "annotation_px": int(annotation_pixel_count),
        }

        cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', fname), cv2.cvtColor(patch_arr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'masks_rgb', fname), cv2.cvtColor(mask_gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'masks_train', fname), mask_train)

        total_patches += 1

    return f"Finished {wsi_name}: {total_patches} patches generated.", slide_metadata


def main():
    wsi_list = sorted(f for f in os.listdir(WSI_PATH) if f.endswith('.tif'))
    print(f"Found {len(wsi_list)} slides to process.")

    max_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Extraction with {max_workers} parallel workers")

    global_metadata = {}
    cnt = 0

    # Parallelizes patch extraction across WSIs (one slide per worker per process).
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_slide, wsi_name): wsi_name for wsi_name in wsi_list}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(wsi_list), desc="Processing WSIs"):
            wsi_name = futures[future]
            try:
                result_msg, slide_metadata = future.result()
                tqdm.write(result_msg)
                if slide_metadata:
                    global_metadata.update(slide_metadata)
            except Exception as exc:
                tqdm.write(f"Slide {wsi_name} generated an exception: {exc}")
                cnt += 1

    json_path = os.path.join(OUTPUT_DIR, 'patches_metadata.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f, indent=4)

    print()
    print(f"Metadata saved to {json_path}.")
    print(f"N patches: {len(global_metadata)}")
    print(f"Error count: {cnt}")


if __name__ == "__main__":
    main()