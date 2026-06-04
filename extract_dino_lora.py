import os
import argparse
import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

# Import your custom modules
from model.dinov3_classifier import DinoV3ClassifierLinearHead
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryPrecision, BinaryRecall
)

# ==========================================
# 1. Define PathologyClassifier (From Script 1)
# ==========================================
class PathologyClassifier(L.LightningModule):
    def __init__(self, config: dict, pos_weight: torch.Tensor = None, all_samples: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weight'])
        self.all_samples = all_samples
        self.config = config

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

        # Loss Function & Metrics (Required to load the checkpoint successfully)
        self.register_buffer('pos_weight', pos_weight)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        metrics = MetricCollection({
            'F1': BinaryF1Score(),
            'AUROC': BinaryAUROC(),
            'Precision': BinaryPrecision(),
            'Recall': BinaryRecall(),
        })
        self.valid_metrics = metrics.clone(prefix='val/')

# ==========================================
# 2. Define Caching Dataset (From Script 2)
# ==========================================
class ColonCacheDataset(Dataset):
    """
    Dataset specifically for caching embeddings. 
    It filters for colon anatomy but keeps ALL frames (no undersampling) 
    so temporal models have continuous sequences.
    """
    def __init__(self, csv_input, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        if isinstance(csv_input, str):
            raw_df = pd.read_csv(csv_input)
        else:
            raw_df = csv_input.copy()

        # Strictly filter for colon and sort by path to maintain temporal order
        self.df = raw_df[raw_df['colon'] == 1].sort_values(by='path').reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        img_name = os.path.join(self.data_root, path)
        
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return both the image and the path (to use as the dictionary key)
        return image, path

# ==========================================
# 3. Helper Functions
# ==========================================
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_transforms(config):
    # Using torchvision.transforms.v2 as requested in Script 1
    return T.Compose([
        T.Resize((config["height"], config["width"])),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ==========================================
# 4. Main Extraction Loop
# ==========================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    print(f"Loading LoRA model checkpoint from: {args.checkpoint}...")
    # Load Lightning module
    lightning_model = PathologyClassifier.load_from_checkpoint(
        args.checkpoint, 
        config=config, 
        pos_weight=torch.tensor([1.0]), 
        strict=False
    )
    lightning_model.to(device)
    lightning_model.eval()

    transform = get_transforms(config)

    # Create datasets using the CSV paths from the config
    train_cache_ds = ColonCacheDataset(config['train_csv'], config['data_root'], transform=transform)
    test_cache_ds = ColonCacheDataset(config['val_csv'], config['data_root'], transform=transform)

    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(train_cache_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_cache_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings_dict = {}

    print(f"Extracting embeddings on {device}...")
    
    with torch.no_grad():
        # --- 1. Process Train Data ---
        compiled_backbone = torch.compile(
            lightning_model.model.backbone, 
            mode="reduce-overhead" 
        )

        # 2. Perform a Warm-up Step
        # torch.compile triggers its JIT compilation on the very first forward pass. 
        # This first pass will take a long time. If you don't do a warm-up, it happens 
        # inside the loop and ruins your tqdm ETA calculation.
        print("Warming up compiled model...")
        dummy_input = torch.randn(batch_size, 3, config["height"], config["width"], device=device)
        _ = compiled_backbone(pixel_values=dummy_input)
        print("Warm-up complete!")

        for images, paths in tqdm(train_loader):
            images = images.to(device)
            
            # 3. Call the COMPILED variable here
            outputs = compiled_backbone(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]
            
            features_cpu = features.cpu()
            for i, path in enumerate(paths):
                embeddings_dict[path] = features_cpu[i].clone()

        # --- 2. Process Test Data ---
        print(f"Processing Testing/Val Data ({len(test_cache_ds)} frames)...")
        for images, paths in tqdm(test_loader):
            images = images.to(device)
            
            # Use the same feature extraction logic here
            outputs = compiled_backbone(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :] 
            
            features_cpu = features.cpu()
            for i, path in enumerate(paths):
                embeddings_dict[path] = features_cpu[i].clone()

    # Save to disk
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(embeddings_dict, args.output)
    print(f"\n✅ Successfully saved {len(embeddings_dict)} final LoRA embeddings to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache LoRA model embeddings to .pt dictionary")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .ckpt file")
    parser.add_argument("--output", type=str, default="/project/lt200353-pcllm/3d_report_gen/CCE/features_dinov3/lora_embeddings_dict.pt", help="Output .pt filename")

    args = parser.parse_args()
    main(args)