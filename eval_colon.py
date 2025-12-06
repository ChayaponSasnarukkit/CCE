import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json  # Added for saving/loading thresholds

# --- Import your specific model components ---
from model.dinov3_classifier import DinoV3ClassifierLinearHead
from datamodule.section import EndoCapsuleDataset
from train_multilabel import PathologyClassifier 

TARGETS = ['ulcer', 'polyp', 'blood', 'erythema',
           'erosion', 'angiectasia', 'IBD', 'hematin', 'lymphangioectasis']

# ==========================================
# Helper: Apply Specific Thresholds
# ==========================================
def apply_thresholds(probs, targets, class_names, thresholds_dict):
    """
    Applies specific thresholds per class and calculates metrics.
    """
    print("\n📉 Applying Loaded Thresholds...")
    results = []
    
    # Containers for global calculation
    all_preds = []
    
    for i, name in enumerate(class_names):
        # 1. Get threshold for this class (default to 0.5 if missing)
        thresh = thresholds_dict.get(name, 0.5)
        
        # 2. Binarize
        y_prob = probs[:, i].numpy()
        y_true = targets[:, i].numpy().astype(int)
        y_pred = (y_prob >= thresh).astype(int)
        
        # 3. Calc Metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        results.append({
            "Class": name,
            "Threshold Used": f"{thresh:.2f}",
            "F1": f"{f1:.4f}",
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}"
        })
    
    return pd.DataFrame(results)

# ==========================================
# Helper: Find Optimal Thresholds
# ==========================================
def find_optimal_thresholds(probs, targets, class_names):
    """
    Iterates through thresholds to find the best F1 score for each class.
    """
    print("\n🔍 Tuning Thresholds to Maximize F1-Score per Class...")
    results = []
    thresholds = np.arange(0.01, 1.00, 0.01)
    optimal_thresholds = {}

    for i, name in enumerate(class_names):
        y_prob = probs[:, i].numpy()
        y_true = targets[:, i].numpy().astype(int)
        
        best_score = -1
        best_thresh = 0.5
        best_metrics = {}

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            if f1 > best_score:
                best_score = f1
                best_thresh = thresh
                best_metrics = {"Precision": precision, "Recall": recall, "F1": f1}

        optimal_thresholds[name] = float(best_thresh) # Ensure it's float for JSON serialization
        
        results.append({
            "Class": name,
            "Best Threshold": f"{best_thresh:.2f}",
            "Max F1": f"{best_metrics['F1']:.4f}",
            "Precision": f"{best_metrics['Precision']:.4f}",
            "Recall": f"{best_metrics['Recall']:.4f}"
        })

    return pd.DataFrame(results), optimal_thresholds

def plot_per_class_metrics(df_results, metric_col="Max F1", save_path="per_class_f1.png"):
    plt.figure(figsize=(12, 6))
    # Handle column naming differences between modes
    y_col = "Max F1" if "Max F1" in df_results.columns else "F1"
    
    # Convert to float for plotting if they are strings
    df_plot = df_results.copy()
    df_plot[y_col] = df_plot[y_col].astype(float)
    
    sns.barplot(data=df_plot, x="Class", y=y_col, hue="Class", palette="viridis", legend=False)
    plt.title(f'Per Class Performance ({y_col})')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Metric plot saved to: {save_path}")
    plt.close()

# ==========================================
# Main Evaluation Function
# ==========================================
def evaluate(ckpt_path, config_path, thresholds_path=None, device_str="cuda"):
    # A. Setup
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Running on {device}")
    
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # B. Load Data
    print("📂 Loading Data...")
    val_transform = T.Compose([
        T.Resize((args["height"], args["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # NOTE: Ensure this points to the correct CSV (Valid vs Test) based on your needs
    # You might want to pass csv_path as an argument too.
    dataset = EndoCapsuleDataset(
        csv_path=args["val_csv_path"], 
        width=args["width"],
        height=args["height"],
        label_names=TARGETS,
        transform=val_transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=False,
        pin_memory=True
    )

    # C. Load Model
    print(f"🔄 Loading Checkpoint: {ckpt_path}")
    model = PathologyClassifier.load_from_checkpoint(
        ckpt_path, config=args, pos_weight=None, all_samples=0, strict=False
    )
    model.to(device)
    model.eval()

    # D. Inference Loop
    all_probs = []
    all_targets = []
    
    print("🚀 Running Inference...")
    with torch.inference_mode():
        for batch in tqdm(loader):
            images, labels = batch
            images = images.to(device)
            logits = model.model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_targets.append(labels)

    all_probs = torch.cat(all_probs).cpu()
    all_targets = torch.cat(all_targets).cpu().int()
    
    # E. Threshold Logic
    print("\n" + "="*40)
    
    if thresholds_path and os.path.exists(thresholds_path):
        # --- MODE 1: Use Provided Thresholds (Test Mode) ---
        print(f"📥 LOADING THRESHOLDS FROM: {thresholds_path}")
        print("="*40)
        
        with open(thresholds_path, 'r') as f:
            loaded_thresholds = json.load(f)
            
        df_results = apply_thresholds(all_probs, all_targets, TARGETS, loaded_thresholds)
        
        print("\n📊 Results using Loaded Thresholds:")
        try:
            print(df_results)
        except ImportError:
            print(df_results)
            
        plot_per_class_metrics(df_results, save_path="test_performance_custom_th.png")

    else:
        # --- MODE 2: Find Optimal Thresholds (Validation Mode) ---
        if thresholds_path:
            print(f"⚠️ Warning: File {thresholds_path} not found. Switching to tuning mode.")
            
        print("🎚️ TUNING OPTIMAL THRESHOLDS (Validation Mode)")
        print("="*40)
        
        df_results, best_thresholds = find_optimal_thresholds(all_probs, all_targets, TARGETS)
        
        print("\n🏆 Best Thresholds Found:")
        try:
            print(df_results.to_markdown(index=False))
        except ImportError:
            print(df_results)
        
        # Save to JSON
        out_json = "optimal_thresholds.json"
        with open(out_json, "w") as f:
            json.dump(best_thresholds, f, indent=4)
        print(f"\n💾 Thresholds saved to {out_json}")
        
        plot_per_class_metrics(df_results, save_path="val_performance_optimized.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--config", type=str, default="./config/pathov1.yaml", help="Path to config file")
    
    # NEW ARGUMENT
    parser.add_argument("--thresholds", type=str, default=None, 
                        help="Path to .json file containing thresholds. If not provided, optimal thresholds will be calculated.")
    
    args = parser.parse_args()
    
    evaluate(args.ckpt, args.config, args.thresholds)