import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from monai.inferers import sliding_window_inference

def evaluate_model(model, test_loader, device, cfg, save_path, output_dir, num_classes=3):
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    class_names = ["Tumor", "Stroma", "Lymphocytes"]
    all_preds_flat = []
    all_labels_flat = []
    
    with torch.no_grad():
        for test_idx, test_data in enumerate(tqdm(test_loader, desc="Testing")):
            test_inputs = test_data["image"].to(device)  
            test_labels = test_data["label"].to(device)

            test_outputs = sliding_window_inference(
                test_inputs, (cfg.dataset.patch_size, cfg.dataset.patch_size), 4, model, mode="gaussian", overlap=0.5
            )
            
            mask = (test_labels != 255)
            preds = torch.argmax(test_outputs, dim=1, keepdim=True)
            labels_clean = torch.where(test_labels == 255, torch.zeros_like(test_labels), test_labels)

            if test_idx < 5: 
                img_cpu = test_inputs[0].cpu().numpy().transpose(1, 2, 0)
                pred_cpu = preds[0, 0].cpu().numpy()
                lbl_cpu = labels_clean[0, 0].cpu().numpy()
                mask_cpu = mask[0, 0].cpu().numpy()
                
                lymph_lbl = (lbl_cpu == 2) & mask_cpu
                lymph_pred = (pred_cpu == 2) & mask_cpu
                
                overlay = img_cpu.copy()
                overlay[lymph_lbl & lymph_pred] = [0, 1, 0] 
                overlay[lymph_pred & ~lymph_lbl] = [1, 0, 0] 
                overlay[lymph_lbl & ~lymph_pred] = [0, 0, 1] 
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(img_cpu)
                plt.subplot(1, 2, 2)
                plt.title("Lymphocyte Mistakes (Red=FP, Blue=FN)")
                plt.imshow(overlay)
                plt.savefig(os.path.join(output_dir, f"error_overlay_lymph_{test_idx}.png"))
                plt.close()

            preds_masked_flat = preds[mask].cpu().numpy().flatten()
            labels_masked_flat = test_labels[mask].cpu().numpy().flatten()
            
            all_preds_flat.append(preds_masked_flat)
            all_labels_flat.append(labels_masked_flat)

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

    test_metrics_dict = {"Confusion_Matrix": cm.tolist()}
    
    print("\nFINAL TEST METRICS:")
    for i, name in enumerate(class_names):
        test_metrics_dict[f"{name}_Dice"] = float(dice[i])
        test_metrics_dict[f"{name}_Jaccard"] = float(jaccard[i])
        test_metrics_dict[f"{name}_Precision"] = float(precision[i])
        test_metrics_dict[f"{name}_Recall"] = float(recall[i])
        
        print(f"{name:<15} | Dice: {dice[i]:.4f} | Jaccard: {jaccard[i]:.4f} | Prec: {precision[i]:.4f} | Recall: {recall[i]:.4f}")
        
    test_metrics_dict["Mean_Dice"] = float(np.mean(dice))
    test_metrics_dict["Mean_Jaccard"] = float(np.mean(jaccard))
    
    print("-" * 65)
    print(f"Mean Test Dice: {np.mean(dice):.4f}")
    print(f"Mean Test Jaccard: {np.mean(jaccard):.4f}\n")

    return test_metrics_dict