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

# --- Import your model definitions ---
# Make sure these are in the python path or same directory
from model.dinov3_classifier import DinoV3ClassifierLinearHead
from train_colon_patho import PathologyClassifier
from datamodule.section import EndoCapsuleDataset

def reshape_transform(tensor, height=14, width=14):
    """
    Converts ViT 1D tokens back to 2D spatial feature maps.
    Handles both standard ViT (1 CLS token) and DINOv2/v3 (1 CLS + 4 Registers).
    """
    # tensor shape: [Batch, Seq_Len, Hidden_Dim]
    seq_len = tensor.shape[1]
    
    # DINOv2/v3: 1 CLS + 4 Registers = 5 non-spatial tokens
    # Standard ViT: 1 CLS = 1 non-spatial token
    
    # We assume the image is square to verify the math
    # If (Seq_Len - 5) is a perfect square, remove 5.
    # If (Seq_Len - 1) is a perfect square, remove 1.
    
    if int(np.sqrt(seq_len - 5)) ** 2 == (seq_len - 5):
        tokens_to_skip = 5
    else:
        tokens_to_skip = 1

    # 1. Remove CLS + Registers
    # Taking the *last* N tokens is safer than slicing from start 
    # because register tokens are usually at the beginning.
    # However, standard convention is they are at the start.
    result = tensor[:, tokens_to_skip:, :] 
    
    # 2. Transpose to [Batch, Hidden, Seq_Len]
    result = result.transpose(1, 2)
    
    # 3. Calculate grid size
    # We recalculate seq_len after dropping tokens
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
        # False Positive: Error=FP
        subset = df[df[col_error] == "FP"]
    elif error_type == "FN":
        # False Negative: Error=FN
        subset = df[df[col_error] == "FN"]
    elif error_type == "TP":
        # True Positive: GT=1, Pred=1
        subset = df[(df[col_gt] == 1) & (df[col_pred] == 1)]
    else:
        return []

    if len(subset) < n:
        return subset['path'].tolist()
    
    return subset.sample(n=n, random_state=42)['path'].tolist()

# ==========================================
# 3. Main Logic
# ==========================================
def run_gradcam(ckpt_path, config_path, csv_path, output_dir="gradcam_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Load Model
    print(f"🔄 Loading Checkpoint: {ckpt_path}")
    import yaml
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # Load Lightning Module
    pl_module = PathologyClassifier.load_from_checkpoint(
        ckpt_path, config=args, pos_weight=None, all_samples=0, strict=False
    )
    pl_module.eval()
    pl_module.cuda()
    
    # Target Layer: For ViT, usually the final LayerNorm of the backbone
    # Structure: pl_module.model (DinoHead) -> .backbone (AutoModel) -> .layernorm
    #target_layers = [pl_module.model.backbone.layernorm]
    target_layers = [pl_module.model.backbone.norm]
    # Initialize GradCAM
    cam = GradCAM(
        model=pl_module.model, 
        target_layers=target_layers, 
        reshape_transform=reshape_transform # CRITICAL FOR VIT
    )

    # B. Load Error Analysis
    df = pd.read_csv(csv_path)
    
    # Identify classes
    prob_cols = [c for c in df.columns if c.endswith('_prob')]
    classes = [c.replace('_prob', '') for c in prob_cols]
    
    # Setup Transform (Must match training preprocessing)
    transform = T.Compose([
        T.Resize((args["height"], args["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n📸 Generatng GradCAMs...")

    for class_name in classes:
        # Get target index for this class
        class_idx = classes.index(class_name)
        
        # Sample images
        samples = {
            "TP": get_samples(df, class_name, "TP", n=2),
            "FP": get_samples(df, class_name, "FP", n=2),
            "FN": get_samples(df, class_name, "FN", n=2)
        }

        for sample_type, paths in samples.items():
            for path in paths:
                # 1. Load Image
                try:
                    # Adjust path logic if relative/absolute paths differ
                    img_path_full = os.path.join("/project/lt200353-pcllm/3d_report_gen/CCE/", path) 
                    # img_path_full = path # Assuming path in CSV is valid
                    
                    pil_img = Image.open(img_path_full).convert('RGB')
                    width, height = pil_img.size
                except Exception as e:
                    print(f"⚠️ Could not load {path}: {e}")
                    continue

                # 2. Preprocess
                input_tensor = transform(pil_img).unsqueeze(0).cuda() # [1, C, H, W]
                
                # 3. Run GradCAM
                # We target the specific class index
                targets = [ClassifierOutputTarget(class_idx)]
                
                # Generate mask
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :] # Take first in batch

                # 4. Visualization
                # Convert PIL to float np array [0,1] for visualization
                rgb_img = np.float32(pil_img.resize((args["width"], args["height"]))) / 255
                
                # Overlay
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                # 5. Save
                filename = os.path.basename(path)
                save_name = f"{class_name}_{sample_type}_{filename}"
                save_full = os.path.join(output_dir, save_name)
                
                cv2.imwrite(save_full, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                print(f"Saved: {save_name}")

    print(f"\n✅ Done! Check the folder: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="./config/pathov1.yaml")
    parser.add_argument("--csv", type=str, default="testerror_analysis.csv")
    
    args = parser.parse_args()
    
    run_gradcam(args.ckpt, args.config, args.csv)
