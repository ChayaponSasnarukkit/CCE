import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassConfusionMatrix
)
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# --- Import your specific model components ---
# Ensure these paths are reachable from where you run the script
from train_section import SectionClassifier
from datamodule.section import EndoCapsuleDataset

# ==========================================
# 2. Evaluation Logic
# ==========================================
def find_optimal_thresholds(probs, targets, class_names):
    """
    Iterates through thresholds to find the best F1 score for each class.
    probs: tensor [N, num_classes]
    targets: tensor [N]
    """
    print("\n🔍 Tuning Thresholds to Maximize F1-Score...")
    results = []
    
    # Range of thresholds to test (0.01 to 0.99)
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    # Store optimal thresholds to return later
    optimal_thresholds = {}

    for i, name in enumerate(class_names):
        # Isolate probabilities for this specific class
        y_prob = probs[:, i].numpy()
        # Create binary ground truth for this class (One-vs-Rest)
        y_true = (targets == i).numpy().astype(int)
        
        best_score = -1
        best_thresh = 0.5
        best_metrics = {}

        # Vectorized search is faster, but loop is clearer for logic
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            # Manual calculation for speed
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            if f1 > best_score:
                best_score = f1
                best_thresh = thresh
                best_metrics = {
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                }

        optimal_thresholds[name] = best_thresh
        
        results.append({
            "Class": name,
            "Best Threshold": f"{best_thresh:.2f}",
            "Max F1": f"{best_metrics['F1']:.4f}",
            "Precision": f"{best_metrics['Precision']:.4f}",
            "Recall": f"{best_metrics['Recall']:.4f}"
        })

    return pd.DataFrame(results), optimal_thresholds

def plot_confusion_matrix(preds, targets, class_names, save_path="confusion_matrix.png"):
    """
    Plots and saves a heatmap of the confusion matrix.
    """
    cm_metric = MulticlassConfusionMatrix(num_classes=len(class_names)).to(preds.device)
    cm = cm_metric(preds, targets).cpu().numpy()
    
    # Normalize by row (True label) to see recall %
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Confusion Matrix saved to: {save_path}")
    plt.close()

# ==========================================
# 3. Main Function
# ==========================================
def evaluate(ckpt_path, config_path, device_str="cuda"):
    # A. Setup
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Running on {device}")
    
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # B. Load Data (Validation Set)
    print("📂 Loading Validation Data...")
    val_transform = T.Compose([
        T.Resize((args["height"], args["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = EndoCapsuleDataset(
        csv_path=args["val_csv_path"],
        width=args["width"],
        height=args["height"],
        label_names=['mouth', 'esophagus', 'stomach', 'small intestine', 'colon'],
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=False,
        pin_memory=True
    )

    # C. Load Model
    print(f"🔄 Loading Checkpoint: {ckpt_path}")
    # We pass strict=False to ignore missing loss weights in checkpoint if any
    model = SectionClassifier.load_from_checkpoint(ckpt_path, config=args, strict=False)
    model.to(device)
    model.eval()

    # D. Inference Loop
    all_probs = []
    all_targets = []
    
    print("🚀 Running Inference...")
    #with torch.no_grad():
    with torch.inference_mode():
        for batch in tqdm(val_loader):
            images, labels = batch
            images = images.to(device)
            
            # Get Logits -> Softmax -> Probabilities
            logits = model.model(images)
            probs = torch.softmax(logits, dim=1)
            
            # Get Target Index
            target_indices = torch.argmax(labels, dim=1)
            
            all_probs.append(probs)
            all_targets.append(target_indices)

    # Concatenate all batches
    all_probs = torch.cat(all_probs).cpu()
    all_targets = torch.cat(all_targets).cpu()
    class_names = val_dataset.label_names
    save_path = "inference_results.pt"

    torch.save({
        'probs': all_probs,
        'targets': all_targets,
        'class_names': class_names
    }, save_path)

    print(f"✅ Results saved to {save_path}")
    # E. 1: Standard Metrics (Argmax)
    print("\n" + "="*40)
    print("📊 STANDARD REPORT (Argmax / Threshold=0.5)")
    print("="*40)
    preds_argmax = torch.argmax(all_probs, dim=1)
    
    # Calculate Macro F1 / Acc
    f1 = MulticlassF1Score(num_classes=5, average='macro')(preds_argmax, all_targets)
    acc = (preds_argmax == all_targets).float().mean()
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1 Score:   {f1:.4f}")

    # Plot Confusion Matrix
    plot_confusion_matrix(preds_argmax, all_targets, class_names, save_path="confusion_matrix_standard.png")

    # E. 2: Threshold Tuning
    print("\n" + "="*40)
    print("🎚️ OPTIMIZED THRESHOLD REPORT")
    print("="*40)
    
    df_results, best_thresholds = find_optimal_thresholds(all_probs, all_targets, class_names)
    
    print("\nResults after Tuning:")
    df_results.to_csv("score.csv")
    print(df_results)
    # Print nice markdown table
    # print(df_results.to_markdown(index=False))
    
    # E. 3: Export Misclassified Examples (Optional suggestion)
    # You could add code here to save filenames of images that were wrong even with best thresholds.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--config", type=str, default="./config/section.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    evaluate(args.ckpt, args.config)
