import os
import torch
import numpy as np
import open3d as o3d
from core.builders import build_network
from utils.runtime_utils import cfg, cfg_from_yaml_file
import matplotlib.pyplot as plt

# Ensure the device is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(cfg_file, ckpt_file):
    """ Loads the model from a checkpoint. """
    cfg_from_yaml_file(cfg_file, cfg)
    net = build_network(cfg)  
    state_dict = torch.load(ckpt_file)
    net.load_state_dict(state_dict['model_state_dict'])
    net = net.to(device)
    net.eval()
    print(f"Model loaded from checkpoint {ckpt_file}")
    return net

def visualize_pointcloud(point_cloud_np):
    """ Visualizes the segmented point cloud using Open3D with color mapping. """
    batch_size = point_cloud_np.shape[0]
    for i in range(batch_size):
        xyz = point_cloud_np[i, :, :3]  # Extract XYZ
        labels = point_cloud_np[i, :, 3]  # Extract Labels

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors = plt.cm.jet(labels / labels.max())[:, :3]  # Color mapping
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])  # Display point cloud

def clear_output_folders(output_dir):
    """ Deletes all existing files in each class folder before creating new ones. """
    for class_id in range(3):  # Assuming classes 0, 1, 2
        class_dir = os.path.join(output_dir, f'class_{class_id}')
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared all files from {class_dir}")
        else:
            os.makedirs(class_dir)  # Ensure the directory exists
            print(f"Created {class_dir}")

def save_predicted_point_clouds(point_cloud_np, output_dir, model_name, combined_class_points):
    """
    Saves each predicted point cloud to separate class folders and accumulates all points for global combination.
    
    combined_class_points (dict): Accumulates points for all clouds in each class across different input files.
    """
    batch_size = point_cloud_np.shape[0]

    for i in range(batch_size):
        xyz = point_cloud_np[i, :, :3]  # Extract XYZ
        labels = point_cloud_np[i, :, 3]  # Extract labels

        for class_id in range(3):  # Assuming classes 0, 1, 2
            class_points = xyz[labels == class_id]

            if len(class_points) > 0:
                class_dir = os.path.join(output_dir, f'class_{class_id}')
                os.makedirs(class_dir, exist_ok=True)

                filename = f"{model_name}_{class_id}.txt"
                file_path = os.path.join(class_dir, filename)

                np.savetxt(file_path, class_points, fmt='%.6f')
                print(f"Saved {len(class_points)} points for class {class_id} to {file_path}")

                # Append to global combined storage
                combined_class_points[class_id].append(class_points)

def process_all_point_clouds(input_dir, output_dir, net):
    """
    Processes all point clouds in the input directory, predicts classes, and saves results.
    Also accumulates all points globally per class and saves a combined file at the end.
    """
    # First, clear existing files in output directories
    clear_output_folders(output_dir)

    point_cloud_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    combined_class_points = {0: [], 1: [], 2: []}  # Dictionary for global storage

    for file_name in point_cloud_files:
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name}...")

        point_cloud_np = net.segment_predict([file_path])  # Predict

        save_predicted_point_clouds(point_cloud_np, output_dir, file_name.split('.')[0], combined_class_points)  # Save results

    # Save globally combined point clouds per class
    for class_id in range(3):
        if combined_class_points[class_id]:  # Ensure there are points to save
            combined_points = np.vstack(combined_class_points[class_id])  # Stack all points together
            combined_filename = os.path.join(output_dir, f'class_{class_id}', f'combined_class_{class_id}.txt')
            np.savetxt(combined_filename, combined_points, fmt='%.6f')
            print(f"Saved globally combined class {class_id} point cloud to {combined_filename}")

# Define directories
input_dir = r"D:\Bonsai\Code\PythonDev\PointStack\To_Predict"
output_dir = r"D:\Bonsai\Code\PythonDev\PointStack\Output_Predicted_Point_Clouds"

# Load the model
cfg_file = r"D:\Bonsai\Code\PythonDev\PointStack\PointStack\experiments\bonsai\Bonsai_HiRes\pointstack.yaml"
ckpt_file = r"D:\Bonsai\Code\PythonDev\PointStack\PointStack\experiments\bonsai\Bonsai_HiRes\ckpt\ckpt-best.pth"
net = load_model(cfg_file, ckpt_file)

# Process all point clouds
process_all_point_clouds(input_dir, output_dir, net)
