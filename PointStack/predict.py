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
    # Load configuration from YAML
    cfg_from_yaml_file(cfg_file, cfg)



    # Build the model
    net = build_network(cfg)  # This builds the network based on the cfg

    # Load checkpoint
    state_dict = torch.load(ckpt_file)
    net.load_state_dict(state_dict['model_state_dict'])
    net = net.to(device)

    # Set the model to evaluation mode
    net.eval()

    print(f"Model loaded from checkpoint {ckpt_file}")
    return net

def visualize_pointcloud(point_cloud_np):
    """ Predicts a point cloud and saves points to class folders based on segmentation. """
    

    
    # Assuming point_cloud_np has shape (batch_size, num_points, 4)
    batch_size = point_cloud_np.shape[0]

    # Loop through each point cloud in the batch
    for i in range(batch_size):
        # Extract XYZ and labels for the current point cloud
        xyz = point_cloud_np[i, :, :3]  # (num_points, 3) for the i-th point cloud
        labels = point_cloud_np[i, :, 3]  # (num_points,) for the i-th point cloud

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Color points by predicted class
        colors = plt.cm.jet(labels / labels.max())[:, :3]  # Jet colormap for the labels
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize point cloud
        o3d.visualization.draw_geometries([pcd])  # Display the i-th point cloud

        


def save_predicted_point_clouds(point_cloud_np, output_dir, model_name):
    """
    Save the points from each predicted class into separate folders.
    
    Args:
        point_cloud_np (np.ndarray): Array of shape (batch_size, num_points, 4), 
                                      where each point is (x, y, z, label).
        output_dir (str): Base directory where class folders will be created.
        model_name (str): Name of the model (used in the filename).
    """
    # Create output directories for each class if they don't exist
    for class_id in range(3):  # Assuming there are 3 classes (0, 1, 2)
        class_dir = os.path.join(output_dir, f'class_{class_id}')
        os.makedirs(class_dir, exist_ok=True)

    batch_size = point_cloud_np.shape[0]

    for i in range(batch_size):
        # Extract the xyz points and labels from the current point cloud
        xyz = point_cloud_np[i, :, :3]  # (num_points, 3)
        labels = point_cloud_np[i, :, 3]  # (num_points,)

        # For each class (0, 1, 2), filter the points that belong to that class
        for class_id in range(3):  # Assuming there are 3 classes (0, 1, 2)
            # Filter points that belong to the current class
            class_points = xyz[labels == class_id]

            # If there are points for the current class, save them
            if len(class_points) > 0:
                # Create the filename for this class and model
                filename = f"{model_name}_{class_id}.txt"
                file_path = os.path.join(output_dir, f'class_{class_id}', filename)

                # Save the points to a .txt file
                np.savetxt(file_path, class_points, fmt='%.6f')

                print(f"Saved {len(class_points)} points for class {class_id} to {file_path}")

# Main function to load all point clouds from a directory and save predictions
def process_all_point_clouds(input_dir, output_dir, net):
    """
    Process all point clouds in the input directory, predict classes, and save them.
    
    Args:
        input_dir (str): Path to the directory containing the point cloud .txt files.
        output_dir (str): Directory where the predicted point clouds will be saved.
        net (nn.Module): The pre-trained model to use for prediction.
    """
    # Get all the point cloud files from the directory
    point_cloud_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    # For each file, load the point cloud, predict, and save the results
    for file_name in point_cloud_files:
        file_path = os.path.join(input_dir, file_name)
        
        print(f"Processing {file_name}...")

        # Predict the point cloud
        point_cloud_np = net.segment_predict([file_path])  # Predict for one point cloud at a time
        
        # Save the predicted points
        save_predicted_point_clouds(point_cloud_np, output_dir, file_name.split('.')[0])  # Save using the file name (without extension)



# Define your input and output directories
input_dir = r"D:\Bonsai\Code\PythonDev\PointStack\To_Predict"
output_dir = r"D:\Bonsai\Code\PythonDev\PointStack\Output_Predicted_Point_Clouds"

# Load the model. It expects a batch size of 2!!
#cfg_file is the .yaml configuration from within the experiments directory for the model being used.
#ckpt_file is the .pth pretrained model from within the experiments directory for the model being used.
cfg_file = r"D:\Bonsai\Code\PythonDev\PointStack\PointStack\experiments\bonsai\Bonsai_HiRes\pointstack.yaml"
ckpt_file = r"D:\Bonsai\Code\PythonDev\PointStack\PointStack\experiments\bonsai\Bonsai_HiRes\ckpt\ckpt-best.pth"
net = load_model(cfg_file, ckpt_file)

# Call the function to process all point clouds in the input directory
process_all_point_clouds(input_dir, output_dir, net)
