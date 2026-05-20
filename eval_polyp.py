import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm
from torchmetrics import MetricCollection
import lightning as L
from datetime import timedelta
import os
from model.dinov3_classifier import DinoV3ClassifierLinearHead
from datamodule.section import EndoCapsuleDataset
import torchvision.transforms.v2 as T
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import (
    BinaryF1Score, 
    BinaryAUROC, 
    BinaryPrecision, 
    BinaryRecall
)
import yaml
import argparse

# Assume EndoscopyTrainDataset and EndoscopyTestDataset are imported from your dataset module
from datamodule.polyp import EndoscopyTrainDataset, EndoscopyTestDataset 

class PathologyClassifier(L.LightningModule):
    def __init__(self, config: dict, pos_weight: torch.Tensor = None, all_samples: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weight'])
        self.all_samples = all_samples
        self.config = config

        # 1. Binary Classification Model (num_classes = 1)
        self.model = DinoV3ClassifierLinearHead(
            num_classes=1, 
            backbone_path=config["backbone_path"],
            freeze_backbone=config["freeze_backbone"],
            use_lora=config["use_lora"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"]
        )

        # 2. Loss Function (Binary Cross Entropy)
        self.register_buffer('pos_weight', pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # 3. Validation Metrics (Updated to Binary)
        metrics = MetricCollection({
            'F1': BinaryF1Score(),
            'AUROC': BinaryAUROC(),
            'Precision': BinaryPrecision(),
            'Recall': BinaryRecall(),
        })
        self.valid_metrics = metrics.clone(prefix='val/')

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.view(-1, 1).float() # Ensure labels are [B, 1] for BCEWithLogits
        
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.view(-1, 1).float()
        
        logits = self.model(images)
        loss = self.criterion(logits, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        preds_proba = torch.sigmoid(logits)
        self.valid_metrics.update(preds_proba, labels.int())

    def on_validation_epoch_end(self):
        metric_dict = self.valid_metrics.compute()
        self.log_dict(metric_dict, logger=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.get("weight_decay", 0.01)
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["lr"])
        
        TOTAL_SAMPLES = int(self.all_samples)
        num_devices = self.config.get("num_devices", 1)
        BATCH_SIZE = self.config["batch_size"] * num_devices
        GRAD_ACCUM_STEPS = self.config["grad_accum_steps"]
        EPOCHS = self.config["epochs"]
        
        total_batches = TOTAL_SAMPLES // BATCH_SIZE
        
        import math
        from transformers import get_cosine_schedule_with_warmup
        num_training_steps = math.ceil(total_batches / GRAD_ACCUM_STEPS) * EPOCHS
        num_warmup_steps = int(num_training_steps * 0.1)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"Using LinearWarmupCosineAnnealingLR: {num_warmup_steps} warmup steps, {num_training_steps} total steps.")
        
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1,
            },
        }

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)

    # --- Transforms ---
    train_transforms = T.Compose([
        T.Resize((config["height"], config["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    val_transforms = T.Compose([
        T.Resize((config["height"], config["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets & DataLoaders ---
    train_dataset = EndoscopyTrainDataset(
        csv_input=config['train_csv'],
        data_root=config['data_root'],
        strategy=config['strategy'],
        ratio=config['ratio'],
        transform=train_transforms,
        undersample_method=config['undersample_method']
    )

    val_dataset = EndoscopyTestDataset(
        csv_input=config['val_csv'],
        data_root=config['data_root'],
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    # --- Calculate Weighted Loss (If Enabled) ---
    pos_weight = None
    if config.get('use_weighted_loss', False):
        # Calculate ratio of negative samples to positive samples in the training set
        num_pos = train_dataset.df['polyp'].sum()
        num_neg = len(train_dataset.df) - num_pos
        weight_val = num_neg / (num_pos + 1e-7)
        pos_weight = torch.tensor([weight_val], dtype=torch.float32)
        print(f"Weighted loss enabled. Positional weight calculated as: {weight_val:.4f}")
    print(f"Loading model from checkpoint: {args.checkpoint}...")
    # strict=False allows loading even if pos_weight isn't exactly matched in the checkpoint
    model = PathologyClassifier.load_from_checkpoint(
        args.checkpoint, 
        config=config, 
        pos_weight=torch.tensor([1.0]), 
        strict=False
    )
    model.to(device)
    model.eval()

    # 4. Run Inference
    all_probs = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating Batches"):
            images = images.to(device)
            
            # Get raw logits and convert to probabilities [0, 1]
            logits = model.model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)

    # 5. Save Results to CSV
    # Because shuffle=False, the dataloader order perfectly matches val_dataset.df
    results_df = val_dataset.df.copy()
    results_df['pred_probability'] = all_probs
    
    # Reorder columns for easier reading
    cols = ['path', 'colon', 'polyp', 'pred_probability']
    results_df = results_df[cols]
    
    results_df.to_csv(args.output, index=False)
    print(f"\n✅ Inference complete! Results saved to {args.output}")
    
    # Quick sanity check print
    print("\n--- Quick Sanity Check ---")
    print(results_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache model inference to CSV")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--output", type=str, default="inference_results.csv", help="Output CSV filename")

    args = parser.parse_args()
    main(args)
