import argparse
import os
import yaml
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm
from datetime import timedelta

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)
from transformers import get_cosine_schedule_with_warmup

# --- Custom Imports (Ensure these are in your path) ---
from model.dinov3_classifier import DinoV3ClassifierLinearHead
from datamodule.section import EndoCapsuleDataset

class SectionClassifier(L.LightningModule):
    def __init__(self, config: dict, class_weights: torch.Tensor = None, loss_type: str = 'focal', all_samples=100):
        """
        Initializes the classification module.
        """
        super().__init__()
        self.save_hyperparameters() # Saves config to the checkpoint
        self.all_samples = all_samples
        self.config = config

        # --- OPTION 5: Backbone Selection ---
        if config.get("backbone") == "resnet":
            import torchvision.models as models
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            if config.get("freeze_backbone", False):
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final fully connected layer for 5 classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 5)
            
        else: # Default to dino
            self.model = DinoV3ClassifierLinearHead(
                num_classes=5,
                backbone_path=config["backbone_path"],
                freeze_backbone=config["freeze_backbone"],
                use_lora=config.get("use_lora", False),
                lora_r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.1),
                target_modules=config.get("target_modules", [])
            )

        # --- Loss Function ---
        self.register_buffer('class_weights', class_weights)

        if loss_type == 'focal':
            # Make sure to move weights to the same device as your model
            # weights = class_weights_tensor.to(device) if class_weights_tensor is not None else None
            self.criterion = MultiClassFocalLoss(alpha_weights=self.class_weights, gamma=2.0)
        else:
            # Standard Cross Entropy
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # --- Validation Metrics ---
        metrics = MetricCollection({
            'F1_macro': MulticlassF1Score(num_classes=5, average='macro'),
            'AUROC_macro': MulticlassAUROC(num_classes=5, average='macro'),
            'Precision_macro': MulticlassPrecision(num_classes=5, average='macro'),
            'Recall_macro': MulticlassRecall(num_classes=5, average='macro'),
        })
        self.valid_metrics = metrics.clone(prefix='val/')

    def training_step(self, batch, batch_idx):
        images, labels = batch
        target_indices = torch.argmax(labels, dim=1)
        logits = self.model(images)
        loss = self.criterion(logits, target_indices)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        target_indices = torch.argmax(labels, dim=1)
        loss = self.criterion(logits, target_indices)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        preds_proba = torch.softmax(logits, dim=1)
        self.valid_metrics.update(preds_proba, target_indices)

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
               'weight_decay': 0.001
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["lr"])
        
        TOTAL_SAMPLES = int(self.all_samples)
        BATCH_SIZE = self.config["batch_size"]
        GRAD_ACCUM_STEPS = self.config["grad_accum_steps"]
        EPOCHS = self.config["epochs"]
        
        total_batches = TOTAL_SAMPLES // BATCH_SIZE
        num_training_steps = math.ceil(total_batches / GRAD_ACCUM_STEPS) * EPOCHS
        num_warmup_steps = int(num_training_steps * 0.1)
        
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        print(f"Using HuggingFace LinearWarmupCosineAnnealingLR with {num_warmup_steps} warmup and {num_training_steps} steps.")
        
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha_weights=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha_weights (Tensor): The class_weights_tensor calculated above.
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(MultiClassFocalLoss, self).__init__()
        self.alpha_weights = alpha_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy loss (incorporating the alpha weights)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha_weights, reduction='none')
        
        # Get the probability of the true class (p_t)
        pt = torch.exp(-ce_loss)
        
        # Apply the focal loss formula: (1 - p_t)^gamma * CE_Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    # --- ARGPARSE SETUP ---
    parser = argparse.ArgumentParser(description="Train EndoCapsule Classifier")
    
    # OPTION 1: Config file path
    parser.add_argument('--config', type=str, default='./config/section.yaml', help='Path to yaml config file')
    # OPTION 2: Savepoint path
    parser.add_argument('--savepoint', type=str, default='/project/lt200353-pcllm/3d_report_gen/CCE/checkpoints', help='Base path for checkpoints')
    # OPTION 3: Weighting
    parser.add_argument('--weighting', type=str, choices=['None', 'standard', 'log-smoothing'], default='standard', help='Class weighting strategy')
    # OPTION 4: Augmentation
    parser.add_argument('--augmentation', type=str, choices=['None', 'augment'], default='augment', help='Toggle training augmentations')
    # OPTION 5: Backbone
    parser.add_argument('--backbone', type=str, choices=['dino', 'resnet'], default='dino', help='Model backbone to use')
    
    parser.add_argument('--loss_type', type=str, choices=['focal', 'bce'], default='focal', help='Loss Type')
    
    cli_args = parser.parse_args()

    # Load Config
    with open(cli_args.config, 'r') as f:
        args = yaml.safe_load(f)
        
    # Inject CLI args into config dictionary to make them globally accessible
    args['savepoint'] = cli_args.savepoint
    args['weighting'] = cli_args.weighting
    args['augmentation'] = cli_args.augmentation
    args['backbone'] = cli_args.backbone

    # --- TRANSFORMS ---
    val_transform = T.Compose([
        T.Resize((args["height"], args["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # OPTION 4: Use val transform if augmentation is 'None'
    if args['augmentation'] == 'augment':
        training_transform = T.Compose([
            T.Resize((args["height"], args["width"])),
            T.RandomApply([
                T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05))
            ], p=0.3),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=15/255, hue=0.05)
            ], p=0.3),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomApply([
                T.GaussianNoise(mean=0.0, sigma=0.05)
            ], p=0.3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        print("⚠️ Augmentation disabled: Using validation transforms for training.")
        training_transform = val_transform

    # --- DATASETS ---
    train_dataset = EndoCapsuleDataset(
        csv_path=args["train_csv_path"],
        width=args["width"],
        height=args["height"],
        label_names=['mouth', 'esophagus', 'stomach', 'small intestine', 'colon'],
        transform=training_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=True
    )

    val_dataset = EndoCapsuleDataset(
        csv_path=args["val_csv_path"],
        width=args["width"],
        height=args["height"],
        label_names=['mouth', 'esophagus', 'stomach', 'small intestine', 'colon'],
        transform=val_transform
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False
    )

    # --- OPTION 3: WEIGHTING ---
    print(f"⚖️ Calculating weights for BCE loss based on training data... (Strategy: {args['weighting']})")
    labels_df = train_dataset.df[['mouth', 'esophagus', 'stomach', 'small intestine', 'colon']]
    label_indices = np.argmax(labels_df.values, axis=1)
    
    class_counts = np.bincount(label_indices, minlength=5)
    total_samples = len(label_indices)
    num_classes = 5
    
    if args['weighting'] == 'None':
        class_weights_tensor = None
        print("✅ Class weights: None (Unweighted)")
    else:
        if args['weighting'] == 'standard':
            weights = total_samples / (num_classes * class_counts + 1e-6)
        elif args['weighting'] == 'log-smoothing':
            # W_c = max(1, log(1 + N_total / (N_classes * N_c)))
            weights = np.maximum(1.0, np.log(1.0 + total_samples / (num_classes * class_counts + 1e-6)))
        elif args['weighting'] == 'focal':
            # Alpha (α) for focal loss: higher weight for rarer classes, bounded between 0 and 1
            frequencies = class_counts / total_samples
            weights = 1.0 - frequencies
        
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

        print("✅ Class weights:")
        for name, weight in zip(['mouth', 'esophagus', 'stomach', 'small intestine', 'colon'], class_weights_tensor):
            print(f"  - {name}: {weight:.4f}")
    
    # Initialize loss function based on args
    
    # --- INITIALIZE MODEL ---
    model = SectionClassifier(args, class_weights=class_weights_tensor, loss_type=args['loss_type'], all_samples=len(train_dataset))

    # --- OPTION 2: SAVEPOINT PATHING ---
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args["savepoint"], "val_best"),
        filename='{epoch:02d}-{val/AUROC_macro:.4f}',
        save_top_k=5,
        monitor='val/AUROC_macro',
        mode='max',
    )

    time_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args["savepoint"], "time"),
        filename='time-based-{epoch:02d}-{step}',
        train_time_interval=timedelta(hours=12),
        save_top_k=-1, 
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger = TensorBoardLogger(
        save_dir=os.path.join(args["savepoint"], 'tb_log'),
        version=''
    )
    
    trainer = L.Trainer(
        max_epochs=args['epochs'],
        accelerator="gpu",
        num_nodes=1,
        devices=-1,
        precision=args['precision'],
        logger=logger,
        callbacks=[val_checkpoint_callback, time_checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=args['gradient_clip_val'],
        accumulate_grad_batches=args['grad_accum_steps'],
    )

    print(f"🚀 Starting training with {args['backbone']} backbone...")
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()
