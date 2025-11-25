import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import collections
from PIL import Image

from sklearn.preprocessing import LabelEncoder

import torch
import torchvision.transforms.v2 as T
training_transform = T.Compose([
    # 1. Resize (A.Resize)
    # T.Resize((height, width)),

    # 2. Geometric Transforms (A.ShiftScaleRotate)
    # T.RandomAffine handles rotation, shift (translate), and scale
    T.RandomApply([
        T.RandomAffine(
            degrees=15,                 # rotate_limit=15
            translate=(0.05, 0.05),     # shift_limit=0.05
            scale=(0.95, 1.05)          # scale_limit=0.05
        )
    ], p=0.3),

    # 3. Color Transforms (A.RGBShift + A.RandomBrightnessContrast)
    # We combine these into one ColorJitter for efficiency.
    # Note: RGBShift is approximated by Hue/Saturation jitter.
    T.RandomApply([
        T.ColorJitter(
            brightness=0.2,   # For BrightnessContrast
            contrast=0.2,     # For BrightnessContrast
            saturation=15/255,# Approx for RGBShift
            hue=0.05          # Approx for RGBShift
        )
    ], p=0.3),

    # 4. Type Conversion (Replaces ToTensorV2 logic)
    # Converts PIL/NumPy -> Image Tensor -> Float32 [0.0, 1.0]
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),

    # 5. Noise (A.GaussNoise) - Native in V2!
    # Note: sigma=0.05 is roughly equivalent to a small var_limit
    T.RandomApply([
        T.GaussianNoise(mean=0.0, sigma=0.05)
    ], p=0.3),

    # 6. Normalize (A.Normalize)
    # T.Normalize expects float tensors in [0, 1] range
    T.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

class EndoCapsuleDataset(Dataset):
    def __init__(self, csv_path, width, height, label_names, transform=None, root_dir="/project/lt200353-pcllm/3d_report_gen/CCE"):
        self.root_dir = root_dir
        self.transform = transform
        self.images_features = []
        self.csv_path = csv_path
        self.width = width
        self.height = height
        self.label_names = label_names

        # filter all frames with unknown != 0
        df = pd.read_csv(csv_path, dtype={'unknown': 'str'})

        self.df = df

        self.labels = self.df[label_names].values.tolist()
                                
    def load_image(self, path):
        try:
            im = Image.open(path)
            if im.mode != 'RGB':
                im = im.convert(mode='RGB')

        except:
            print(f'IMAGE FAILED TO LOAD: {path}')
            exit()
        return im

    def __getitem__(self, index):

        labels = torch.tensor(self.labels[index], dtype=torch.float)

        image_path = os.path.join(self.root_dir, os.path.normpath(str(self.df.iloc[index]['path'])))

        image = self.load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, labels

    def __len__(self) -> int:
        return len(self.labels)
