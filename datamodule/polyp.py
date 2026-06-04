import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class WindowedPolypDatasetv2(Dataset):
    def __init__(self, csv_input, embeddings_dict, window_size=32, label_col='polyp', 
                 is_train=True, 
                 apply_undersample=False, undersample_ratio=1.0, undersample_method='framerate',
                 apply_oversample=False, oversample_ratio=1.0):
        
        self.window_size = window_size
        self.embeddings_dict = embeddings_dict
        self.label_col = label_col
        self.is_train = is_train
        
        # 1. Load CSV and extract video ID
        if isinstance(csv_input, str):
            raw_df = pd.read_csv(csv_input)
        else:
            raw_df = csv_input.copy()
            
        raw_df['video_id'] = raw_df['path'].apply(lambda x: str(x).split('/')[0])
        
        raw_df['frame_num'] = raw_df['path'].str.extract(r'(\d+)').astype(int)
        
        # 2. Store FULL sequential videos for window extraction (SORT BY NUMERIC FRAME)
        full_colon_df = raw_df[raw_df['colon'] == 1].sort_values(by=['video_id', 'frame_num']).reset_index(drop=True)
        self.videos = [v_df.reset_index(drop=True) for _, v_df in full_colon_df.groupby('video_id')]
        self.video_id_to_idx = {v_df['video_id'].iloc[0]: i for i, v_df in enumerate(self.videos)}

        # 3. Handle Sampling Strategies (Training Only)
        if self.is_train:
            # Separate into classes
            normals_df = full_colon_df[full_colon_df['polyp'] == 0].reset_index(drop=True)
            polyps_df = full_colon_df[full_colon_df['polyp'] == 1].reset_index(drop=True)
            
            # --- STEP A: UNDERSAMPLING ---
            if apply_undersample:
                target_normal_count = int(len(polyps_df) * undersample_ratio)
                if target_normal_count < len(normals_df):
                    if undersample_method == 'random':
                        normals_df = normals_df.sample(n=target_normal_count, random_state=42)
                    elif undersample_method == 'framerate':
                        indices = np.linspace(0, len(normals_df) - 1, target_normal_count).astype(int)
                        normals_df = normals_df.iloc[indices]
                print(f"Undersampling applied. Normal frames reduced to: {len(normals_df)}")
            
            # --- STEP B: OVERSAMPLING ---
            if apply_oversample:
                # Calculate how many polyps we need to hit the ratio compared to current normals
                target_polyp_count = int(len(normals_df) * oversample_ratio)
                if target_polyp_count > len(polyps_df):
                    # Sample WITH replacement to duplicate polyps
                    polyps_df = polyps_df.sample(n=target_polyp_count, replace=True, random_state=42)
                print(f"Oversampling applied. Polyp frames inflated to: {len(polyps_df)}")

            # Combine and shuffle
            sampled_targets = pd.concat([normals_df, polyps_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        else:
            # Validation/Testing Mode: Keep untouched distribution
            sampled_targets = full_colon_df

        # 4. Map targets to flat indices
        self.flat_indices = []
        for _, row in sampled_targets.iterrows():
            v_id = row['video_id']
            p_path = row['path']
            v_idx = self.video_id_to_idx[v_id]
            f_idx = self.videos[v_idx][self.videos[v_idx]['path'] == p_path].index[0]
            
            self.flat_indices.append((v_idx, f_idx))
            
        print(f"Dataset active targets: {len(self.flat_indices)} frames.")

    def __len__(self):
        return len(self.flat_indices)

    def __getitem__(self, idx):
        v_idx, f_idx = self.flat_indices[idx]
        v_df = self.videos[v_idx]
        num_frames = len(v_df)
        
        half_left = self.window_size // 2
        half_right = self.window_size - half_left
        
        window_indices = [
            max(0, min(j, num_frames - 1)) 
            for j in range(f_idx - half_left, f_idx + half_right)
        ]
        
        paths = v_df['path'].values[window_indices]
        labels = v_df[self.label_col].values[window_indices]
        window_embeddings = [self.embeddings_dict[p].clone() for p in paths]
        
        return torch.stack(window_embeddings), torch.tensor(labels, dtype=torch.long)
    
class WindowedPolypDataset(Dataset):
    def __init__(self, csv_input, embeddings_dict, window_size=32, label_col='polyp', 
                 apply_undersample=False, strategy=1, ratio=1.0, undersample_method='framerate'):
        self.window_size = window_size
        self.embeddings_dict = embeddings_dict
        self.label_col = label_col
        
        # 1. Load CSV
        if isinstance(csv_input, str):
            raw_df = pd.read_csv(csv_input)
        else:
            raw_df = csv_input.copy()
            
        # Dynamically infer video folder ID
        raw_df['video_id'] = raw_df['path'].apply(lambda x: str(x).split('/')[0])
        
        # 2. Store the FULL sequential videos for perfect temporal context retrieval
        full_colon_df = raw_df[raw_df['colon'] == 1].sort_values(by=['video_id', 'path']).reset_index(drop=True)
        self.videos = [v_df.reset_index(drop=True) for _, v_df in full_colon_df.groupby('video_id')]
        self.video_id_to_idx = {v_df['video_id'].iloc[0]: i for i, v_df in enumerate(self.videos)}

        # 3. Determine the Valid Target Prediction Points
        if apply_undersample:
            print(f"Applying Undersampling Strategy {strategy} (Ratio: {ratio}, Method: {undersample_method})...")
            if strategy == 1:
                sampled_targets = self._strategy_1(raw_df, ratio)
            elif strategy == 2:
                sampled_targets = self._strategy_2(raw_df, ratio)
            elif strategy == 3:
                sampled_targets = self._strategy_3(raw_df, ratio, undersample_method)
            else:
                raise ValueError("Strategy must be 1, 2, or 3")
        else:
            print("No undersampling applied. Using ALL available colon frames.")
            sampled_targets = full_colon_df  # Uses every single frame

        # 4. Map selected targets to their exact positions inside the full videos
        self.flat_indices = []
        for _, row in sampled_targets.iterrows():
            v_id = row['video_id']
            p_path = row['path']
            v_idx = self.video_id_to_idx[v_id]
            v_df = self.videos[v_idx]
            f_idx = v_df[v_df['path'] == p_path].index[0]
            
            self.flat_indices.append((v_idx, f_idx))
            
        print(f"Dataset active targets: {len(self.flat_indices)} frames.")

    def _strategy_1(self, df, ratio):
        colon_df = df[df['colon'] == 1]
        polyps = colon_df[colon_df['polyp'] == 1]
        normals = colon_df[colon_df['polyp'] == 0]
        target_count = min(int(len(polyps) * ratio), len(normals))
        sampled_normals = normals.sample(n=target_count, random_state=42)
        return pd.concat([polyps, sampled_normals])

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
        return pd.concat([polyps, sampled_normals])

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
        return pd.concat([all_polyps, sampled_normals])

    def __len__(self):
        return len(self.flat_indices)

    def __getitem__(self, idx):
        v_idx, f_idx = self.flat_indices[idx]
        v_df = self.videos[v_idx]
        num_frames = len(v_df)
        
        half_left = self.window_size // 2
        half_right = self.window_size - half_left
        
        # Edge-clamping retrieves frames smoothly from the uncut sequential array
        window_indices = [
            max(0, min(j, num_frames - 1)) 
            for j in range(f_idx - half_left, f_idx + half_right)
        ]
        
        paths = v_df['path'].values[window_indices]
        labels = v_df[self.label_col].values[window_indices]
        
        window_embeddings = [self.embeddings_dict[p].clone() for p in paths]
        
        return torch.stack(window_embeddings), torch.tensor(labels, dtype=torch.long)

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