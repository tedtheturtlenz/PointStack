import os
import json
import random

def split_files(input_dir, output_dir, train_ratio=0.7, test_ratio=0.1, val_ratio=0.2):
    # Ensure the sum of the ratios is 1
    assert train_ratio + test_ratio + val_ratio == 1, "Ratios must sum to 1"

    # List to hold file paths for each split
    train_files = []
    test_files = []
    val_files = []
    
    # Walk through the directories
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        
        if os.path.isdir(folder_path):
            # Get all .txt files in the folder
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            
            # Shuffle the file list
            random.shuffle(txt_files)
            
            # Calculate split sizes
            total_files = len(txt_files)
            train_size = int(total_files * train_ratio)
            test_size = int(total_files * test_ratio)
            val_size = total_files - train_size - test_size  # Ensure all files are assigned
            
            # Split the files into train, test, and validation, remove .txt extension
            train_files.extend([f"shape_data/{folder}/{os.path.splitext(f)[0]}" for f in txt_files[:train_size]])
            test_files.extend([f"shape_data/{folder}/{os.path.splitext(f)[0]}" for f in txt_files[train_size:train_size + test_size]])
            val_files.extend([f"shape_data/{folder}/{os.path.splitext(f)[0]}" for f in txt_files[train_size + test_size:]])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write the file lists to JSON files in the specified output directory
    with open(os.path.join(output_dir, 'shuffled_train_file_list.json'), 'w') as f:
        json.dump(train_files, f, indent=4)
    
    with open(os.path.join(output_dir, 'shuffled_test_file_list.json'), 'w') as f:
        json.dump(test_files, f, indent=4)
    
    with open(os.path.join(output_dir, 'shuffled_val_file_list.json'), 'w') as f:
        json.dump(val_files, f, indent=4)

# Example usage:
input_folder = r"D:\Bonsai\Code\PythonDev\PointStack\PointStack\data\bonsai"  # Replace this with your actual folder path
output_folder = r"D:\Bonsai\Code\PythonDev\PointStack\PointStack\data\bonsai\train_test_split"
split_files(input_folder, output_folder)
