
# from model import InterpolatedPositionalEmbedding3D
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
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)
import lightning as L
from datetime import timedelta
import os
from model.dinov3_classifier import DinoV3ClassifierLinearHead
from datamodule.section import EndoCapsuleDataset
import torchvision.transforms.v2 as T
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

class SectionClassifier(L.LightningModule):
    def __init__(self, config: dict, class_weights: torch.Tensor = None, all_samples=100):
        """
        Initializes the classification module.
        Args:
            config (dict): A configuration dictionary containing model, optimizer,
                           and scheduler parameters.
        """
        super().__init__()
        self.save_hyperparameters() # Saves config to the checkpoint
        self.all_samples = all_samples
        self.config = config

        self.model = DinoV3ClassifierLinearHead(
            num_classes=5,
            backbone_path=config["backbone_path"],
            freeze_backbone = config["freeze_backbone"],
            use_lora = config["use_lora"],
            lora_r = config["lora_r"],
            lora_alpha = config["lora_alpha"],
            lora_dropout = config["lora_dropout"],
            target_modules= config["target_modules"]
        )

        # --- 2. Loss Function ---
        self.register_buffer('class_weights', class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # --- 4. Validation Metrics ---
        metrics = MetricCollection({
            'F1_macro': MulticlassF1Score(num_labels=5, average='macro'),
            'AUROC_macro': MulticlassAUROC(num_labels=5, average='macro'),
            'Precision_macro': MulticlassPrecision(num_labels=5, average='macro'),
            'Recall_macro': MulticlassRecall(num_labels=5, average='macro'),
        })
        self.valid_metrics = metrics.clone(prefix='val/')

    def training_step(self, batch, batch_idx):
        #torch.autograd.set_detect_anomaly(True)
        images, labels = batch
        target_indices = torch.argmax(labels, dim=1)
        logits = self.model(images)
        loss = self.criterion(logits, target_indices)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step, calculating loss and updating metrics.
        """
        images, labels = batch
        logits = self.model(images)
        target_indices = torch.argmax(labels, dim=1)
        loss = self.criterion(logits, target_indices)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        preds_proba = torch.softmax(logits, dim=1)
        self.valid_metrics.update(preds_proba, target_indices)

    def on_validation_epoch_end(self):
        """
        Computes and logs aggregated metrics at the end of the validation epoch.
        """
        metric_dict = self.valid_metrics.compute()
        self.log_dict(metric_dict, logger=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        """
        Sets up the AdamW optimizer and a learning rate scheduler.
        """
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
               'weight_decay': 0.001  # e.g. 0.01
            },
            {
                'params': [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]

        # Create the optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = self.config["lr"])
        TOTAL_SAMPLES = int(self.all_samples//4)
        BATCH_SIZE = self.config["batch_size"]
        GRAD_ACCUM_STEPS = self.config["grad_accum_steps"]
        EPOCHS = self.config["epoch"]
        total_batches = TOTAL_SAMPLES // BATCH_SIZE
        # Use ceiling division to not miss the last partial step, if any
        import math
        from transformers import get_cosine_schedule_with_warmup
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
                "interval": "step", # CORRECTED: This scheduler should update per step
                "frequency": 1,
            },
        }

import yaml
def main():
    """Main function to load config and run training."""
    # 1. Load Config
    with open('./config/section.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    training_transform = T.Compose([
        # 1. Resize (A.Resize)
        T.Resize((args["height"], args["width"])),

        # 2. Geometric Transforms (A.ShiftScaleRotate)
        T.RandomApply([
            T.RandomAffine(
                degrees=15,                 # rotate_limit=15
                translate=(0.05, 0.05),     # shift_limit=0.05
                scale=(0.95, 1.05)          # scale_limit=0.05
            )
        ], p=0.3),

        # 3. Color Transforms (A.RGBShift + A.RandomBrightnessContrast)
        T.RandomApply([
            T.ColorJitter(
                brightness=0.2,   # For BrightnessContrast
                contrast=0.2,     # For BrightnessContrast
                saturation=15/255,# Approx for RGBShift
                hue=0.05          # Approx for RGBShift
            )
        ], p=0.3),

        # 4. Converts PIL/NumPy -> Image Tensor -> Float32 [0.0, 1.0]
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),

        # 5. Guassian Noise 
        T.RandomApply([
            T.GaussianNoise(mean=0.0, sigma=0.05)
        ], p=0.3),

        # 6. Normalize using value from processor config
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

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

    val_transform = T.Compose([
        # 1. Resize (A.Resize)
        T.Resize((args["height"], args["width"])),

        # 2. Converts PIL/NumPy -> Image Tensor -> Float32 [0.0, 1.0]
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),

        # 3. Normalize using value from processor config
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

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

    print("⚖️ Calculating weights for BCE loss based on training data distribution...")
    labels_df = train_dataset.df[['mouth', 'esophagus', 'stomach', 'small intestine', 'colon']]

    label_indices = np.argmax(labels_df.values, axis=1)
    
    # Count occurrences of each class
    class_counts = np.bincount(label_indices, minlength=5)
    total_samples = len(label_indices)
    num_classes = 5
    
    # Calculate weights: Total / (Num_Classes * Count)
    # This balances the contribution of each class to the loss
    weights = total_samples / (num_classes * class_counts + 1e-6)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("✅ Class weights:")
    for name, weight in zip(['mouth', 'esophagus', 'stomach', 'small intestine', 'colon'], class_weights_tensor):
        print(f"  - {name}: {weight:.2f}")
    
    model = SectionClassifier(args, pclass_weights=class_weights_tensor, all_samples=len(train_dataset))
    print(model)

    val_checkpoint_callback = ModelCheckpoint(
        dirpath="/project/lt200353-pcllm/3d_report_gen/CCE/checkpoints/val_best",
        filename='{epoch:02d}-{val/AUROC_macro:.4f}',
        save_top_k=5,
        monitor='val/AUROC_macro',
        mode='max',
    )

    time_checkpoint_callback = ModelCheckpoint(
        dirpath="/project/lt200353-pcllm/3d_report_gen/CCE/checkpoints/time",
        filename='time-based-{epoch:02d}-{step}',
        train_time_interval=timedelta(hours=12),
        save_top_k=-1, # Saves all time-based checkpoints
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # 6. Initialize Trainer
    logger = TensorBoardLogger(
        save_dir='./tb_log',
        version=''
    )
    trainer = L.Trainer(
        max_epochs=args['epochs'],
        accelerator="gpu",
        num_nodes=1,
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        precision=args['precision'],
        logger=logger,
        callbacks=[val_checkpoint_callback, time_checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=args['gradient_clip_val'],
        accumulate_grad_batches=args['grad_accum_steps'],
    )

    trainer.validate(model, val_dataloader)
    # 7. Start Training
    print("🚀 Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
