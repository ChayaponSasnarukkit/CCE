import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# --- Import your specific model components ---
from train_section import SectionClassifier
from datamodule.section import EndoCapsuleDataset


# ==========================================
# 1. Helper Functions
# ==========================================
def get_dataloader(csv_path, args, transform):
    """Helper to instantiate dataset and dataloader."""
    dataset = EndoCapsuleDataset(
        csv_path=csv_path,
        width=args["width"],
        height=args["height"],
        label_names=['mouth', 'esophagus', 'stomach', 'small intestine', 'colon'],
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=False,
        pin_memory=True
    )
    return loader, dataset.label_names

def run_inference(model, loader, device, desc="Inference"):
    """Runs a dataloader through the model and returns all probabilities and targets."""
    all_probs = []
    all_targets = []
    print(f"🚀 Running {desc}...")
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc):
            images, labels = batch
            images = images.to(device)
            
            logits = model.model(images)
            probs = torch.softmax(logits, dim=1)
            target_indices = torch.argmax(labels, dim=1)
            
            all_probs.append(probs.cpu())
            all_targets.append(target_indices.cpu())
            
    return torch.cat(all_probs), torch.cat(all_targets)

# ==========================================
# 2. Evaluation & Tuning Logic
# ==========================================
def find_optimal_thresholds(probs, targets, class_names):
    """
    Treats multi-class as One-vs-Rest to find the optimal F1 threshold per class.
    probs: tensor [N, num_classes]
    targets: tensor [N]
    """
    print("\n🔍 Tuning Thresholds on Validation Set (Maximizing F1)...")
    probs_np = probs.numpy()
    targets_np = targets.numpy()
    
    optimal_thresholds = []
    
    for i, class_name in enumerate(class_names):
        binary_targets = (targets_np == i).astype(int)
        class_probs = probs_np[:, i]
        
        # Scikit-learn PR curve for exact thresholding
        precision, recall, thresholds = precision_recall_curve(binary_targets, class_probs)
        
        # Calculate F1 for all thresholds
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        optimal_thresholds.append(best_threshold)
        
        print(f"{class_name.capitalize():<16}: {best_threshold:.4f} (Val Max F1: {f1_scores[best_idx]:.4f})")
        
    return np.array(optimal_thresholds)

def plot_confusion_matrix(preds, targets, class_names, save_path="confusion_matrix.png", title="Normalized Confusion Matrix"):
    """Plots and saves a heatmap of the confusion matrix using sklearn."""
    cm = confusion_matrix(targets.numpy(), preds.numpy())
    
    # Normalize by row (True label) to see recall %
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Confusion Matrix saved to: {save_path}")
    plt.close()

def get_per_class_accuracy(cm):
    """Calculates strict per-class accuracy: (TP + TN) / Total"""
    accs = []
    total = cm.sum()
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        accs.append((tp + tn) / total)
    return accs

# ==========================================
# 3. Main Function
# ==========================================
def evaluate(cli_args, device_str="cuda"):
    # A. Setup
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Running on {device}")
    
    os.makedirs(cli_args.base_output_dir, exist_ok=True)
    print(f"📂 Saving all outputs to: {cli_args.base_output_dir}")

    with open(cli_args.config, 'r') as f:
        args = yaml.safe_load(f)

    args['backbone'] = cli_args.backbone

    val_transform = T.Compose([
        T.Resize((args["height"], args["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # B. Load Data (Validation AND Test Sets)
    print("📂 Loading Datasets...")
    # NOTE: Ensure args["val_csv_path"] exists in your config!
    val_csv = args.get("val_csv_path", args["test_csv_path"].replace("test", "val"))
    val_loader, class_names = get_dataloader(val_csv, args, val_transform)
    test_loader, _ = get_dataloader(args["test_csv_path"], args, val_transform)

    # C. Load Model
    print(f"🔄 Loading Checkpoint: {cli_args.ckpt}")
    model = SectionClassifier.load_from_checkpoint(cli_args.ckpt, config=args, strict=False)
    model.to(device)
    model.eval()

    # D. Step 1: Validation Inference & Threshold Tuning
    val_probs, val_targets = run_inference(model, val_loader, device, desc="Validation Phase")
    optimal_thresholds = find_optimal_thresholds(val_probs, val_targets, class_names)
    
    # E. Step 2: Test Set Inference
    test_probs, test_targets = run_inference(model, test_loader, device, desc="Test Phase")
    
    # Save raw test probabilities for future reference
    pt_save_path = os.path.join(cli_args.base_output_dir, "test_inference_results.pt")
    torch.save({'probs': test_probs, 'targets': test_targets, 'class_names': class_names}, pt_save_path)

    # F. Generate Predictions
    # 1. Baseline Predictions (Standard Argmax)
    baseline_preds = torch.argmax(test_probs, dim=1)
    
    # 2. Tuned Predictions (Scaled Argmax)
    # Convert numpy array to tensor to avoid warnings
    thresholds_tensor = torch.tensor(optimal_thresholds, dtype=test_probs.dtype)
    scaled_probs = test_probs / thresholds_tensor
    tuned_preds = torch.argmax(scaled_probs, dim=1)

    # G. Calculate Metrics & Export to CSV
    print("\n📊 Generating Final Reports on Test Set...")
    
    # Get scikit-learn dictionaries
    target_np = test_targets.numpy()
    base_report = classification_report(target_np, baseline_preds.numpy(), target_names=class_names, output_dict=True)
    tuned_report = classification_report(target_np, tuned_preds.numpy(), target_names=class_names, output_dict=True)
    
    # Get confusion matrices for exact per-class accuracy (TP + TN / Total)
    cm_base = confusion_matrix(target_np, baseline_preds.numpy())
    cm_tuned = confusion_matrix(target_np, tuned_preds.numpy())
    
    base_accs = get_per_class_accuracy(cm_base)
    tuned_accs = get_per_class_accuracy(cm_tuned)

    # Compile data for CSV
    csv_rows = []
    for i, cls in enumerate(class_names):
        csv_rows.append({
            "Class": cls,
            "Support": base_report[cls]["support"],
            "Optimal Threshold": optimal_thresholds[i],
            
            "Base Accuracy": base_accs[i],
            "Base Precision": base_report[cls]["precision"],
            "Base Recall": base_report[cls]["recall"],
            "Base F1": base_report[cls]["f1-score"],
            
            "Tuned Accuracy": tuned_accs[i],
            "Tuned Precision": tuned_report[cls]["precision"],
            "Tuned Recall": tuned_report[cls]["recall"],
            "Tuned F1": tuned_report[cls]["f1-score"],
        })

    df_scores = pd.DataFrame(csv_rows)
    
    # Save CSV
    csv_save_path = os.path.join(cli_args.base_output_dir, "score.csv")
    df_scores.to_csv(csv_save_path, index=False)
    
    print(f"\n✅ All metrics saved to {csv_save_path}")
    print(df_scores.to_string(index=False))
    
    # Plot both confusion matrices
    plot_confusion_matrix(baseline_preds, test_targets, class_names, 
                          save_path=os.path.join(cli_args.base_output_dir, "cm_baseline.png"),
                          title="Baseline Confusion Matrix (Argmax)")
                          
    plot_confusion_matrix(tuned_preds, test_targets, class_names, 
                          save_path=os.path.join(cli_args.base_output_dir, "cm_tuned.png"),
                          title="Tuned Confusion Matrix (Thresholds applied)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--config", type=str, default="./config/section.yaml", help="Path to config file")
    parser.add_argument("--backbone", type=str, choices=['dino', 'resnet'], default='dino')
    parser.add_argument("--base_output_dir", type=str, default="./", help="Base directory to save output files")
    
    cli_args = parser.parse_args()
    evaluate(cli_args)
