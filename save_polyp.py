import os
import shutil
import pandas as pd
from tqdm import tqdm  # Optional: provides a nice progress bar

def copy_polyp_colon_frames(csv_path, src_root, dest_root):
    """
    Reads a CSV, filters for polyp and colon frames, and copies them to a new directory.
    
    Args:
        csv_path (str): Path to your dataset CSV file.
        src_root (str): The base directory where your original images live.
        dest_root (str): The target directory to copy the filtered images to.
    """
    # 1. Read the CSV
    df = pd.read_csv(csv_path)
    
    # 2. Filter the dataframe
    # Assuming "polyp and colon frames" means frames where BOTH conditions are true (1)
    filtered_df = df[(df['colon'] == 1) & (df['polyp'] == 1)]
    
    print(f"Found {len(filtered_df)} frames matching the criteria out of {len(df)} total frames.")
    
    # Ensure the base destination directory exists
    os.makedirs(dest_root, exist_ok=True)
    
    # 3. Iterate through the filtered list and copy files
    missing_files = 0
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Copying files"):
        # Construct full paths
        src_path = os.path.join(src_root, str(row['path']))
        dest_path = os.path.join(dest_root, str(row['path']))
        
        # Create any necessary subdirectories in the destination
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"\nWarning: File not found -> {src_path}")
            missing_files += 1
            
    print("\n--- Copy Complete ---")
    print(f"Successfully copied: {len(filtered_df) - missing_files}")
    if missing_files > 0:
        print(f"Files missing/failed: {missing_files}")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Define your paths here
    CSV_FILE = "/project/lt200353-pcllm/3d_report_gen/CCE/train_polyp.csv"
    DEST_DIR = "./train_polyp/"
    SOURCE_DIR = "/project/lt200353-pcllm/3d_report_gen/CCE/"
    
    copy_polyp_colon_frames(CSV_FILE, SOURCE_DIR, DEST_DIR)
