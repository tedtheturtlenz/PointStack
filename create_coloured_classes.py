import os
import numpy as np
import open3d as o3d

def load_colored_point_clouds(input_dir):
    """
    Loads all original point clouds with color (x, y, z, r, g, b) into a dictionary for quick lookup.

    Returns:
        original_data (dict): { (x, y, z) : (r, g, b) }
    """
    original_data = {}

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_dir, file_name)
            print(f"Loading original point cloud: {file_name}")

            # Load the file (assuming space-separated values)
            point_data = np.loadtxt(file_path)
            
            # Check if the format is (x, y, z, r, g, b, nx, ny, nz)
            if point_data.shape[1] >= 6:
                for row in point_data:
                    xyz = tuple(row[:3])  # (x, y, z)
                    rgb = tuple(row[3:6])  # (r, g, b)
                    original_data[xyz] = rgb  # Store in dictionary

    return original_data

def add_color_to_combined_clouds(combined_dir, original_data, output_dir):
    """
    Matches each point in the combined class clouds with its RGB values from the original data.
    Saves new colored versions in both .txt and .ply formats in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_id in range(3):  # Assuming classes 0, 1, 2
        combined_file = os.path.join(combined_dir, f'class_{class_id}', f'combined_class_{class_id}.txt')
        
        if os.path.exists(combined_file):
            print(f"Processing {combined_file}...")

            # Load the combined point cloud (x, y, z)
            combined_points = np.loadtxt(combined_file)

            # Store new points with color
            colored_points = []

            for point in combined_points:
                xyz = tuple(point[:3])  # Extract (x, y, z)

                # Find corresponding color in original data
                if xyz in original_data:
                    r, g, b = original_data[xyz]
                    colored_points.append([*xyz, r, g, b])  # Append (x, y, z, r, g, b)
                else:
                    print(f"Warning: No color found for {xyz}, skipping...")

            # Save new colored point cloud in .ply format
            if colored_points:
                formatted_data = np.array(colored_points)
                
                # Create an Open3D PointCloud object
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(formatted_data[:, :3])  # Points (x, y, z)
                pcd.colors = o3d.utility.Vector3dVector(formatted_data[:, 3:] / 255.0)  # Colors (r, g, b), normalize to [0, 1]

                # Save to .ply format
                output_ply = os.path.join(output_dir, f'colored_combined_class_{class_id}.ply')
                o3d.io.write_point_cloud(output_ply, pcd)

                print(f"Saved colored combined class {class_id} to {output_ply}")

# Define your directories
original_input_dir = r"D:/Bonsai/Code/PythonDev/PointStack/To_Predict_Colour"
combined_input_dir = r"D:/Bonsai/Code/PythonDev/PointStack/Output_Predicted_Point_Clouds"
output_colored_dir = r"D:/Bonsai/Code/PythonDev/PointStack/Output_Predicted_Point_Clouds/Colored_Combined"

# Load original point clouds with color data
original_point_data = load_colored_point_clouds(original_input_dir)

# Match and save colored versions of combined point clouds
add_color_to_combined_clouds(combined_input_dir, original_point_data, output_colored_dir)
