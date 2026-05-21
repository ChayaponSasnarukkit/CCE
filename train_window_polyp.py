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

from model.temporal_classifier import TemporalWindowClassifier
from datamodule.polyp import WindowedPolypDataset
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
    # Handle edge case where a batch might theoretically only have 1 class
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auroc, auprc = 0.0, 0.0

    # Negative Predictive Value (NPV)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "NPV": npv,
        "AUROC": auroc,
        "AUPRC": auprc
    }

# Setup Config Parameters
config = {
    "window_size": 32,
    "embed_dim": 1024,
    "batch_size": 128,
    "epochs": 15,
    "lr": 1e-4,
    "warmup_epochs": 2, # How many epochs to linearly scale LR before cosine decay
    "checkpoint_dir": "/project/lt200353-pcllm/3d_report_gen/CCE/checkpoints/448i_strat2",
    
    # --- Imbalance Settings ---
    "undersample": {
        "active": True,            
        "strategy": 2,             
        "ratio": 3.0,              
        "method": "framerate"      
    },
    "loss_type": "weighted_ce",
    "class_frequencies": [300000, 600]
}

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, 2), targets: (B,)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss) # computes probability of target class
        
        # Apply alpha weighting factor dynamically based on class target
        alpha_t = self.alpha * (targets == 1).float() + (1 - self.alpha) * (targets == 0).float()
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
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
    train_dataset = WindowedPolypDataset(
        csv_input=train_csv_path,
        embeddings_dict=embeddings_dict,
        window_size=config["window_size"],
        apply_undersample=config["undersample"]["active"],
        strategy=config["undersample"]["strategy"],
        ratio=config["undersample"]["ratio"],
        undersample_method=config["undersample"]["method"]
    )
    
    val_dataset = WindowedPolypDataset(
        csv_input=val_csv_path,
        embeddings_dict=embeddings_dict,
        window_size=config["window_size"],
        apply_undersample=False # Force False for validation
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # 2. Build Model & Loss
    model = TemporalWindowClassifier(window_size=config["window_size"], embed_dim=config["embed_dim"], num_heads=16, depth=4).to(device)

    if config["loss_type"] == "weighted_ce":
        # 1. Convert frequencies to tensor
        freqs = torch.tensor(config["class_frequencies"], dtype=torch.float32)
        
        # 2. Calculate inverse frequencies
        inverse_freqs = 1.0 / freqs
        
        # 3. Normalize so the sum of weights equals the number of classes (2)
        # This keeps the overall gradient scale stable and predictable
        normalized_weights = inverse_freqs / inverse_freqs.sum() * len(freqs)
        
        criterion = nn.CrossEntropyLoss(weight=normalized_weights.to(device))
        
        # For [300k, 600], this will print something like: [0.0039, 1.9960]
        print(f"Loss Configured: Weighted CE with normalized weights: {normalized_weights.tolist()}")
    elif config["loss_type"] == "focal":
        # Alpha controls class weighting. 0.75 pushes it to focus heavily on class 1 (polyps)
        criterion = FocalLoss(alpha=0.75, gamma=2.0) 
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    # 3. Learning Rate Scheduler (Linear Warmup -> Cosine Annealing)
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
            logits = model(features)
            loss = criterion(logits, target_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step() # Step the scheduler per batch for smooth warmup
            
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
                
                logits = model(features)
                loss = criterion(logits, target_labels)
                val_loss += loss.item()
                
                # Get probabilities for the positive class (index 1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target_labels.cpu().numpy())

        val_loss /= len(val_loader)
        
        # Calculate metrics using numpy arrays
        metrics = calculate_metrics(
            np.array(all_targets), 
            np.array(all_probs), 
            np.array(all_preds)
        )

        # Print Epoch Summary
        print(f"\nEpoch [{epoch+1}/{config['epochs']}] | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}  | AUPRC: {metrics['AUPRC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}  | NPV:   {metrics['NPV']:.4f}")
        print(f"AUROC:     {metrics['AUROC']:.4f}  | Acc:   {metrics['Accuracy']:.4f}")

        # ==========================
        #      CHECKPOINTING
        # ==========================
        # Primary goal: Maximize Recall. Tie-breaker: Maximize AUPRC.
        # Note: We enforce a minimum AUPRC to prevent it from saving a model that just guesses '1' for everything.
        
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
    TRAIN_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/train_polyp.csv" # Make sure to point to your actual CSV files
    TEST_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/val_test_polyp.csv"
    OUTPUT_FILE = os.path.join(DATA_ROOT, "features_dinov3", "224_colon_embeddings_dict.pt")

    print(config, OUTPUT_FILE, flush=True)
    run_pipeline(train_csv_path=TRAIN_CSV, val_csv_path=TEST_CSV, embeddings_dict_path=OUTPUT_FILE)
