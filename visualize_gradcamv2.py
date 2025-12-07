import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms.v2 as T
from PIL import Image
import yaml

# --- Import your model definitions ---
# Make sure these are in the python path or same directory
from model.dinov3_classifier import DinoV3ClassifierLinearHead
from train_colon_patho import PathologyClassifier
from datamodule.section import EndoCapsuleDataset

# ==========================================
# 1. ViT Reshape Transform (Corrected for DINOv3)
# ==========================================
def reshape_transform(tensor, height=14, width=14):
    """
    Converts ViT 1D tokens back to 2D spatial feature maps.
    Handles both standard ViT (1 CLS token) and DINOv2/v3 (1 CLS + 4 Registers).
    """
    # tensor shape: [Batch, Seq_Len, Hidden_Dim]
    seq_len = tensor.shape[1]
    
    # We assume the image is square to verify the math
    # If (Seq_Len - 5) is a perfect square, remove 5 (CLS + 4 Regs).
    # If (Seq_Len - 1) is a perfect square, remove 1 (CLS).
    if int(np.sqrt(seq_len - 5)) ** 2 == (seq_len - 5):
        tokens_to_skip = 5
    else:
        tokens_to_skip = 1

    # 1. Remove non-spatial tokens
    result = tensor[:, tokens_to_skip:, :] 
    
    # 2. Transpose to [Batch, Hidden, Seq_Len]
    result = result.transpose(1, 2)
    
    # 3. Calculate grid size
    spatial_seq_len = result.size(2)
    grid_size = int(np.sqrt(spatial_seq_len))
    
    # 4. Reshape to [Batch, Hidden, Grid, Grid]
    result = result.reshape(tensor.size(0), tensor.size(2), grid_size, grid_size)
    
    return result

# ==========================================
# 2. Helper: Sample Images
# ==========================================
def get_samples(df, class_name, error_type, n=2):
    """
    Filters the CSV for specific errors (FP, FN) or Correct (TP)
    """
    col_gt = f"{class_name}_gt"
    col_pred = f"{class_name}_pred"
    col_error = f"{class_name}_error"
    
    if error_type == "FP":
        subset = df[df[col_error] == "FP"]
    elif error_type == "FN":
        subset = df[df[col_error] == "FN"]
    elif error_type == "TP":
        subset = df[(df[col_gt] == 1) & (df[col_pred] == 1)]
    else:
        return []

    if len(subset) < n:
        return subset['path'].tolist()
    
    # Add random_state for reproducibility
    return subset.sample(n=n, random_state=42)['path'].tolist()

# ==========================================
# 3. Main Logic
# ==========================================
def run_gradcam(ckpt_path, config_path, csv_path, output_dir="gradcam_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Load Model
    print(f"🔄 Loading Checkpoint: {ckpt_path}")
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # Load Lightning Module
    pl_module = PathologyClassifier.load_from_checkpoint(
        ckpt_path, config=args, pos_weight=None, all_samples=0, strict=False
    )
    pl_module.eval()
    pl_module.cuda()
    
    # 1. Unwrap LoRA if present
    if hasattr(pl_module.model.backbone, "base_model"):
        vit = pl_module.model.backbone.base_model.model
    else:
        vit = pl_module.model.backbone

    # 2. Target Layer: norm1 of the LAST block (Crucial for ViT Heatmaps)
    target_layers = [vit.layer[-1].norm1]
    print(f"🎯 Target Layer found: {target_layers[0]}")

    # Initialize GradCAM with the inner model
    cam = GradCAM(
        model=pl_module.model, 
        target_layers=target_layers, 
        reshape_transform=reshape_transform 
    )

    # B. Load Error Analysis
    df = pd.read_csv(csv_path)
    
    # Identify classes
    prob_cols = [c for c in df.columns if c.endswith('_prob')]
    classes = [c.replace('_prob', '') for c in prob_cols]
    
    # Setup Transform
    transform = T.Compose([
        T.Resize((args["height"], args["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n📸 Generating interpretable GradCAM plots in: {output_dir}")

    for class_name in classes:
        class_idx = classes.index(class_name)
        
        samples = {
            "TP": get_samples(df, class_name, "TP", n=2),
            "FP": get_samples(df, class_name, "FP", n=2),
            "FN": get_samples(df, class_name, "FN", n=2)
        }

        for sample_type, paths in samples.items():
            for path in paths:
                # 1. Load & Preprocess
                try:
                    img_path_full = os.path.join("/project/lt200353-pcllm/3d_report_gen/CCE/", path) 
                    pil_img = Image.open(img_path_full).convert('RGB')
                except Exception as e:
                    print(f"⚠️ Could not load {path}: {e}")
                    continue

                input_tensor = transform(pil_img).unsqueeze(0).cuda()
                
                # 2. Run GradCAM
                targets = [ClassifierOutputTarget(class_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

                # 3. Prepare Images for Plotting
                # Original image resized for visualization
                rgb_img = np.float32(pil_img.resize((args["width"], args["height"]))) / 255
                # GradCAM overlay
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                # 4. Create Interpretable Plot (Side-by-Side)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Left Plot: Original
                axes[0].imshow(rgb_img)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # Right Plot: GradCAM
                axes[1].imshow(visualization)
                axes[1].set_title(f"GradCAM Activation ({class_name})")
                axes[1].axis('off')
                
                # Determine Labels
                gt_label = "Positive" if sample_type in ["TP", "FN"] else "Negative"
                pred_label = "Positive" if sample_type in ["TP", "FP"] else "Negative"
                filename = os.path.basename(path)
                
                # Main Title with all info
                plt.suptitle(f"File: {filename}\nTarget: {class_name} | Type: {sample_type}\nGround Truth: {gt_label} | Prediction: {pred_label}", 
                             fontsize=14, fontweight='bold', y=1.02)
                
                # 5. Save Plot
                save_name = f"{class_name}_{sample_type}_{filename.split('.')[0]}.png"
                save_full = os.path.join(output_dir, save_name)
                
                plt.tight_layout()
                plt.savefig(save_full, bbox_inches='tight')
                plt.close(fig) # Free memory
                print(f"Saved plot: {save_full}")

    print(f"\n✅ Done! Check the folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="./config/pathov1.yaml")
    parser.add_argument("--csv", type=str, default="testerror_analysis.csv")
    
    args = parser.parse_args()
    
    run_gradcam(args.ckpt, args.config, args.csv)
