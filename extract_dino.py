import os
import torch
import torch.nn as nn
from transformers import AutoModel
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 1. Define Backbone (DINOv3)
# ==========================================
class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        print(f"Loading model: {model_id} ...")
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        # Extract the CLS token feature
        return outputs.last_hidden_state[:, 0, :]

def get_dinov3_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ==========================================
# 2. Define Dataset
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
# 3. Main Extraction Loop
# ==========================================
def extract_and_cache_embeddings(train_csv, test_csv, data_root, model, transform, save_path, batch_size=64, num_workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create datasets
    train_cache_ds = ColonCacheDataset(train_csv, data_root, transform=transform)
    test_cache_ds = ColonCacheDataset(test_csv, data_root, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_cache_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_cache_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings_dict = {}

    print(f"Extracting embeddings on {device}...")
    
    with torch.no_grad(): # Essential for saving memory during inference
        # 1. Process Train Data
        print(f"Processing Training Data ({len(train_cache_ds)} frames)...")
        for images, paths in tqdm(train_loader):
            images = images.to(device)
            features = model(images) 
            features_cpu = features.cpu()
            
            for i, path in enumerate(paths):
                embeddings_dict[path] = features_cpu[i].clone()

        # 2. Process Test Data
        print(f"Processing Testing Data ({len(test_cache_ds)} frames)...")
        for images, paths in tqdm(test_loader):
            images = images.to(device)
            features = model(images)
            features_cpu = features.cpu()
            
            for i, path in enumerate(paths):
                embeddings_dict[path] = features_cpu[i].clone()

    # Save to disk
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(embeddings_dict, save_path)
    print(f"Successfully saved {len(embeddings_dict)} embeddings to {save_path}")


# ==========================================
# Example Usage
# ==========================================
# train_csv: "/project/lt200353-pcllm/3d_report_gen/CCE/train_polyp.csv"
# val_csv: "/project/lt200353-pcllm/3d_report_gen/CCE/val_test_polyp.csv"
# data_root: "/project/lt200353-pcllm/3d_report_gen/CCE/"
if __name__ == "__main__":
    # Define paths
    DATA_ROOT = "/project/lt200353-pcllm/3d_report_gen/CCE/"
    TRAIN_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/train_polyp.csv" # Make sure to point to your actual CSV files
    TEST_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/val_test_polyp.csv"
    OUTPUT_FILE = os.path.join(DATA_ROOT, "features_dinov3", "224_colon_embeddings_dict.pt")
    
    MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    BATCH_SIZE = 256
    NUM_WORKERS = 8

    # 1. Define your transformation for DINOv3
    val_transform = get_dinov3_transform()

    # 2. Load the DINOv3 feature extractor
    feature_extractor = DINOFeatureExtractor(MODEL_ID)
    
    # 3. Run the extraction
    extract_and_cache_embeddings(
        train_csv=TRAIN_CSV, 
        test_csv=TEST_CSV, 
        data_root=DATA_ROOT, 
        model=feature_extractor, 
        transform=val_transform,
        save_path=OUTPUT_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )