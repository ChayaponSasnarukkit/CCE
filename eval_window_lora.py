import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from tqdm import tqdm

from model.temporal_classifier import TemporalWindowClassifier
from torch.utils.data import Dataset

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
                target_polyp_count = int(len(normals_df) * oversample_ratio)
                if target_polyp_count > len(polyps_df):
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
        
        # Get the path of the specific target frame
        target_path = v_df['path'].values[f_idx]
        
        return torch.stack(window_embeddings), torch.tensor(labels, dtype=torch.long), target_path

def calculate_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auroc, auprc = 0.0, 0.0

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

config = {
    "window_size": 4,
    "embed_dim": 1024,
    "batch_size": 128,
    "checkpoint_dir": "/project/lt200353-pcllm/3d_report_gen/CCE/checkpoints/window_4_ratio_3v3",
    "checkpoint_name": "best_model.pth", 
    "output_csv": "lora_evaluation_predictionsv3.csv"
}

def run_evaluation(test_csv_path, embeddings_dict_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Embeddings
    print("Loading pre-computed embeddings into memory...", flush=True)
    embeddings_dict = torch.load(embeddings_dict_path, map_location='cpu')
    print("LOADING COMPLETE", flush=True)

    # 2. Initialize Test Dataset
    test_dataset = WindowedPolypDatasetv2(
        csv_input=test_csv_path,
        embeddings_dict=embeddings_dict,
        window_size=config["window_size"],
        is_train=False,         
        apply_undersample=False,
        apply_oversample=False
    )

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # 3. Build Model & Load Checkpoint
    model = TemporalWindowClassifier(
        window_size=config["window_size"], 
        embed_dim=config["embed_dim"], 
        num_heads=16, 
        depth=4,
        num_classes=1,
        drop_rate=0.1, drop_path_rate=0.1
    ).to(device)
    
    checkpoint_path = os.path.join(config["checkpoint_dir"], config["checkpoint_name"])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with Validation Metrics: {checkpoint.get('metrics', 'N/A')}")

    # 4. Evaluation Loop
    model.eval()
    all_preds = []
    all_probs_pos = []
    all_probs_neg = []
    all_targets = []
    all_paths = [] 
    
    mid_idx = config["window_size"] // 2

    print(f"--- Starting Evaluation on {device} ---", flush=True)
    test_pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for features, labels, paths in test_pbar:
            features, labels = features.to(device), labels.to(device)
            target_labels = labels[:, mid_idx]
            
            logits = model(features)
            
            # Flatten to [batch_size] to ensure correct shape for sigmoid and arrays
            logits = logits.view(-1)
            
            # Calculate probabilities using Sigmoid for single-class BCE
            probs_pos = torch.sigmoid(logits)
            probs_neg = 1.0 - probs_pos
            
            # Threshold at 0.5 for final binary prediction
            preds = (probs_pos >= 0.5).int()
            
            all_probs_pos.extend(probs_pos.cpu().numpy())
            all_probs_neg.extend(probs_neg.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_labels.cpu().numpy())
            
            # Paths is a tuple of strings from dataloader
            all_paths.extend(paths)

    # 5. Calculate Metrics
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs_pos)
    y_pred = np.array(all_preds)
    
    metrics = calculate_metrics(y_true, y_prob, y_pred)
    
    print("\n================ EVALUATION RESULTS ================")
    print(f"Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"Precision:   {metrics['Precision']:.4f}")
    print(f"Recall:      {metrics['Recall']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"AUROC:       {metrics['AUROC']:.4f}")
    print(f"AUPRC:       {metrics['AUPRC']:.4f}")
    print("====================================================")

    # 6. Save to CSV
    output_df = pd.DataFrame({
        "Path": all_paths, 
        "True_Label": y_true,
        "Predicted_Class": y_pred,
        "Probability_Class_0 (Normal)": all_probs_neg,
        "Probability_Class_1 (Polyp)": y_prob
    })
    
    output_path = os.path.join(config["checkpoint_dir"], config["output_csv"])
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved all predictions and probabilities to: {output_path}")

if __name__ == '__main__':
    DATA_ROOT = "/project/lt200353-pcllm/3d_report_gen/CCE/"
    TEST_CSV = "/project/lt200353-pcllm/3d_report_gen/CCE/val_test_polyp.csv"
    EMBEDDINGS_FILE = os.path.join(DATA_ROOT, "features_dinov3", "lora_embeddings_dict.pt")

    run_evaluation(test_csv_path=TEST_CSV, embeddings_dict_path=EMBEDDINGS_FILE)
