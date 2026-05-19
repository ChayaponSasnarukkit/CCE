import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class EndoscopyTrainDataset(Dataset):
    """
    Training Dataset that applies sampling strategies to handle class imbalance 
    and increase polyp diversity.
    """
    def __init__(self, csv_input, data_root, strategy=3, ratio=1.0, transform=None, undersample_method='framerate'):
        """
        Args:
            csv_input (str or pd.DataFrame): Path to the csv file or dataframe.
            data_root (str): Directory with all the images.
            strategy (int): 1, 2, or 3 corresponding to your requested options.
            ratio (float): Ratio of normal frames to polyp frames.
            transform (callable, optional): Optional transform to be applied on a sample.
            undersample_method (str): 'framerate' or 'random' (used in Strategy 3).
        """
        self.data_root = data_root
        self.transform = transform
        
        # Accept either a path string or a pre-loaded DataFrame
        if isinstance(csv_input, str):
            raw_df = pd.read_csv(csv_input)
        else:
            raw_df = csv_input.copy()

        # Apply the chosen strategy to build the final training list
        if strategy == 1:
            print("only colon frame, random undersampling")
            self.df = self._strategy_1(raw_df, ratio)
        elif strategy == 2:
            print("only colon frame, framerate undersampling")
            self.df = self._strategy_2(raw_df, ratio)
        elif strategy == 3:
            print(f"all polyp from all anatomy, but normal frames are come from colon only, {undersample_method} undersampling")
            self.df = self._strategy_3(raw_df, ratio, undersample_method)
        else:
            raise ValueError("Strategy must be 1, 2, or 3")

    def _strategy_1(self, df, ratio):
        colon_df = df[df['colon'] == 1]
        polyps = colon_df[colon_df['polyp'] == 1]
        normals = colon_df[colon_df['polyp'] == 0]
        
        target_count = min(int(len(polyps) * ratio), len(normals))
        sampled_normals = normals.sample(n=target_count, random_state=42)
        
        return pd.concat([polyps, sampled_normals]).sample(frac=1, random_state=42).reset_index(drop=True)

    def _strategy_2(self, df, ratio):
        colon_df = df[df['colon'] == 1]
        polyps = colon_df[colon_df['polyp'] == 1]
        normals = colon_df[colon_df['polyp'] == 0].sort_values(by='path').reset_index(drop=True)
        
        target_count = min(int(len(polyps) * ratio), len(normals))
        
        if target_count > 0:
            indices = np.linspace(0, len(normals) - 1, target_count).astype(int)
            sampled_normals = normals.iloc[indices]
        else:
            sampled_normals = pd.DataFrame(columns=normals.columns)
            
        return pd.concat([polyps, sampled_normals]).sample(frac=1, random_state=42).reset_index(drop=True)

    def _strategy_3(self, df, ratio, method):
        polyps_colon = df[(df['colon'] == 1) & (df['polyp'] == 1)]
        polyps_other = df[(df['colon'] == 0) & (df['polyp'] == 1)]
        all_polyps = pd.concat([polyps_colon, polyps_other])
        
        normals_colon = df[(df['colon'] == 1) & (df['polyp'] == 0)].sort_values(by='path').reset_index(drop=True)
        target_count = min(int(len(all_polyps) * ratio), len(normals_colon))
        
        if method == 'random':
            sampled_normals = normals_colon.sample(n=target_count, random_state=42)
        elif method == 'framerate':
            if target_count > 0:
                indices = np.linspace(0, len(normals_colon) - 1, target_count).astype(int)
                sampled_normals = normals_colon.iloc[indices]
            else:
                sampled_normals = pd.DataFrame(columns=normals_colon.columns)
                
        return pd.concat([all_polyps, sampled_normals]).sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_root, self.df.iloc[idx]['path'])
        
        # Load image and convert to RGB (to ensure consistency across grayscale/RGBA images)
        image = Image.open(img_name).convert('RGB')
        
        # Assuming binary classification for polyp detection (0 or 1)
        label = torch.tensor(self.df.iloc[idx]['polyp'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


class EndoscopyTestDataset(Dataset):
    """
    Testing Dataset that strict filters for colon anatomy but maintains 
    the untouched, real-world class distribution.
    """
    def __init__(self, csv_input, data_root, transform=None):
        """
        Args:
            csv_input (str or pd.DataFrame): Path to the csv file or dataframe.
            data_root (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        self.transform = transform
        
        if isinstance(csv_input, str):
            raw_df = pd.read_csv(csv_input)
        else:
            raw_df = csv_input.copy()

        # Filter strictly for target anatomy (colon) and preserve natural order
        self.df = raw_df[raw_df['colon'] == 1].sort_values(by='path').reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_root, self.df.iloc[idx]['path'])
        
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.df.iloc[idx]['polyp'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label