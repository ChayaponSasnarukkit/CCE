import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
import torch.nn.functional as F

from model.temporal_classifier import TemporalWindowClassifier
from datamodule.polyp import WindowedPolypDatasetv2
from tqdm import tqdm

def calculate_metrics(y_true, y_prob, y_pred):
    """
    y_true: 1D numpy array of true binary labels
    y_prob: 1D numpy array of probabilities for the positive class (polyp=1)
    y_pred: 1D numpy array of hard predictions (0 or 1)
    """
    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Area Under the Curves
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auroc, auprc = 0.0, 0.0

    # Negative Predictive Value (NPV) & Sensitivity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "sensitivity": sensitivity,
        "AUROC": auroc,
        "AUPRC": auprc
    }

# Setup Config Parameters
config = {
    "window_size": 16,
    "embed_dim": 1024,
    "batch_size": 128,
    "epochs": 15,
    "lr": 3e-5,
    "warmup_epochs": 2, 
    "checkpoint_dir": "/project/lt200353-pcllm/3d_report_gen/CCE/checkpoints/window_16_ratio_3",
    
    # --- Imbalance Settings ---
    "undersample": {
        "active": True,           
        "ratio": 3,              
        #"method": "random",
        "method": "framerate"
    },
    "oversample": {
        "active": False,            
        "ratio": 0.333             
    },
    # Changed loss options to reflect BCE
    "loss_type": "weighted_bce", 
    #"class_frequencies": [300000, 300000]
    "class_frequencies": [300000, 60000] # [Negatives, Positives]
}

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B,), targets: (B,) float
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) # computes probability of target class
        
        # Apply alpha weighting factor dynamically based on class target
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def run_pipeline(train_csv_path, val_csv_path, embeddings_dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # 1. Load Embeddings and Datasets
    print("Loading pre-computed embeddings into memory...", flush=True)
    embeddings_dict = torch.load(embeddings_dict_path, map_location='cpu')
    print("LOADING COMPLETE", flush=True)

    train_dataset = WindowedPolypDatasetv2(
        csv_input=train_csv_path,
        embeddings_dict=embeddings_dict,
        window_size=config["window_size"],
        is_train=True,
        apply_undersample=config["undersample"]["active"],
        undersample_ratio=config["undersample"]["ratio"],
        undersample_method=config["undersample"]["method"],
        apply_oversample=config["oversample"]["active"],
        oversample_ratio=config["oversample"]["ratio"]
    )

    val_dataset = WindowedPolypDatasetv2(
        csv_input=val_csv_path,
        embeddings_dict=embeddings_dict,
        window_size=config["window_size"],
        is_train=False,         
        apply_undersample=False,
        apply_oversample=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # 2. Build Model & Loss
    model = TemporalWindowClassifier(
        window_size=config["window_size"], 
        embed_dim=config["embed_dim"], 
        num_heads=8, 
        depth=4,
        num_classes=1,
        drop_rate=0.1, drop_path_rate=0.1
    ).to(device)

    # Note: TemporalWindowClassifier must output (B, 1) for BCE!

    if config["loss_type"] == "weighted_bce":
        # BCE pos_weight is calculated as: number of negative samples / number of positive samples
        neg_samples = config["class_frequencies"][0]
        pos_samples = config["class_frequencies"][1]
        pos_weight = torch.tensor([neg_samples / pos_samples], dtype=torch.float32).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss Configured: Weighted BCE with pos_weight: {pos_weight.item():.2f}")
        
    elif config["loss_type"] == "focal":
        criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0) 
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    # 3. Learning Rate Scheduler
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # Tracking variables
    best_recall = 0.0
    best_auprc = 0.0
    _best_recall = 0.0
    mid_idx = config["window_size"] // 2

    # 4. Main Training Engine
    print(f"--- Starting Training on {device} ---", flush=True)
    for epoch in range(config["epochs"]):
        
        # ==========================
        #        TRAINING
        # ==========================
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)
        
        for features, labels in train_pbar:
            features, labels = features.to(device), labels.to(device)
            target_labels = labels[:, mid_idx]
            
            optimizer.zero_grad()
            
            # Squeeze output to (B,) if model outputs (B, 1)
            logits = model(features).squeeze(-1) 
            
            # BCE expects targets to be floats
            loss = criterion(logits, target_labels.float())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step() 
            
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # ==========================
        #       VALIDATION
        # ==========================
        model.eval()
        val_loss = 0.0
        all_preds, all_probs, all_targets = [], [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=False)
        with torch.no_grad():
            for features, labels in val_pbar:
                features, labels = features.to(device), labels.to(device)
                target_labels = labels[:, mid_idx]
                
                logits = model(features).squeeze(-1)
                loss = criterion(logits, target_labels.float())
                val_loss += loss.item()
                
                # BCE evaluation uses Sigmoid, and thresholding at 0.5
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())

        val_loss /= len(val_loader)
        
        metrics = calculate_metrics(
            np.array(all_targets), 
            np.array(all_probs), 
            np.array(all_preds)
        )

        # Print Epoch Summary
        print(f"\nEpoch [{epoch+1}/{config['epochs']}] | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}  | AUPRC: {metrics['AUPRC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}  | Sensitivity:   {metrics['sensitivity']:.4f}")
        print(f"AUROC:     {metrics['AUROC']:.4f}  | Acc:   {metrics['Accuracy']:.4f}")

        # ==========================
        #      CHECKPOINTING
        # ==========================
        is_best = False
        is_recall_best = False
        if metrics['Recall'] > best_recall and metrics['AUPRC'] > 0.01:
            is_best = True
        elif metrics['Recall'] == best_recall and metrics['AUPRC'] > best_auprc:
            is_best = True

        if metrics['Recall'] > _best_recall:
            is_recall_best = True

        if is_best:
            best_recall = metrics['Recall']
            best_auprc = metrics['AUPRC']
            save_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            torch.save(checkpoint, save_path)
            print(f"⭐ New best model saved! (Recall: {best_recall:.4f}, AUPRC: {best_auprc:.4f})")

        if is_recall_best:
            _best_recall = metrics['Recall']
            save_path = os.path.join(config["checkpoint_dir"], "_best_model.pth")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            torch.save(checkpoint, save_path)
            print(f"⭐ New best model saved! (Recall: {_best_recall:.4f}")

if __name__ == '__main__':
    DATA_ROOT = "/project/lt200353-pcllm/3d_report_gen/CCE/"
    TRAIN_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/train_polyp.csv" 
    TEST_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/val_test_polyp.csv"
    #OUTPUT_FILE = os.path.join(DATA_ROOT, "features_dinov3", "224_colon_embeddings_dict.pt")
    OUTPUT_FILE = "/project/lt200353-pcllm/3d_report_gen/CCE/features_dinov3/lora_embeddings_dict.pt"
    print(config, OUTPUT_FILE, flush=True)
    run_pipeline(train_csv_path=TRAIN_CSV, val_csv_path=TEST_CSV, embeddings_dict_path=OUTPUT_FILE)
