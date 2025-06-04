import matplotlib

import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def png_to_pointcloud(png_path, fx, fy, cx, cy, scale=1000.0, max_points=8192, z_max=0.2):
    # Load 16-bit PNG
    depth = cv2.imread(png_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = depth / scale

    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    mask = (z > 0) & (z <= z_max)  # Filter both zero depth and high Z values
    points = np.stack((x[mask], y[mask], z[mask]), axis=-1)
    
    # Downsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return points  # shape: [N, 3]


def apply_transformation(points, transformation_matrix):
    """Apply 4x4 transformation matrix to 3D points"""
    # Convert points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])  # [N, 4]
    
    # Apply transformation
    transformed_homo = points_homo @ transformation_matrix.T  # [N, 4]
    
    # Convert back to 3D coordinates
    transformed = transformed_homo[:, :3]
    return transformed


def visualize_point_cloud_registration(src_points, tgt_points, transformation_matrix, 
                                     title="Point Cloud Registration", save_path=None,
                                     subsample=1000):
    """
    Visualize source, target, and transformed source point clouds
    
    Args:
        src_points: Source point cloud [N, 3] numpy array
        tgt_points: Target point cloud [M, 3] numpy array  
        transformation_matrix: 4x4 transformation matrix numpy array
        title: Plot title
        save_path: Path to save the plot (optional)
        subsample: Number of points to display (for performance)
    """
    # Subsample points for visualization if too many
    if len(src_points) > subsample:
        src_idx = np.random.choice(len(src_points), subsample, replace=False)
        src_vis = src_points[src_idx]
    else:
        src_vis = src_points
        
    if len(tgt_points) > subsample:
        tgt_idx = np.random.choice(len(tgt_points), subsample, replace=False)
        tgt_vis = tgt_points[tgt_idx]
    else:
        tgt_vis = tgt_points
    
    # Apply transformation to source points
    src_transformed = apply_transformation(src_vis, transformation_matrix)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: All three point clouds
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(src_vis[:, 0], src_vis[:, 1], src_vis[:, 2], 
               c='red', s=1, alpha=0.6, label='Source (Red)')
    ax1.scatter(tgt_vis[:, 0], tgt_vis[:, 1], tgt_vis[:, 2], 
               c='green', s=1, alpha=0.6, label='Target (Green)')
    ax1.scatter(src_transformed[:, 0], src_transformed[:, 1], src_transformed[:, 2], 
               c='blue', s=1, alpha=0.6, label='Transformed Source (Blue)')
    ax1.set_title('All Point Clouds')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Transformed Source vs Target
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(src_transformed[:, 0], src_transformed[:, 1], src_transformed[:, 2], 
               c='blue', s=1, alpha=0.8, label='Transformed Source (Blue)')
    ax2.scatter(tgt_vis[:, 0], tgt_vis[:, 1], tgt_vis[:, 2], 
               c='green', s=1, alpha=0.8, label='Target (Green)')
    ax2.set_title('After Registration')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def visualize_dataset_pair(dataset, idx, model=None, device='cpu'):
    """
    Visualize a specific pair from the dataset
    
    Args:
        dataset: PNGPointCloudDataset instance
        idx: Index of the pair to visualize
        model: Trained model (optional, for showing predicted transformation)
        device: Device for model inference
    """
    # Get the point cloud pair
    src, tgt, gt_pose = dataset[idx]
    
    # Convert to numpy
    src_np = src.numpy()
    tgt_np = tgt.numpy()
    gt_pose_np = gt_pose.numpy()
    
    print(f"Visualizing pair {idx}: {dataset.png_paths[idx]} -> {dataset.png_paths[idx+1]}")
    print(f"Source points: {src_np.shape[0]}")
    print(f"Target points: {tgt_np.shape[0]}")
    
    # If model is provided, get predicted transformation
    if model is not None:
        model.eval()
        with torch.no_grad():
            # Add batch dimension
            src_batch = src.unsqueeze(0).to(device)
            tgt_batch = tgt.unsqueeze(0).to(device)
            
            # Get prediction
            pred_pose = model(src_batch, tgt_batch)
            pred_pose_np = pred_pose.cpu().numpy()[0]
            
            # Show predicted transformation
            visualize_point_cloud_registration(
                src_np, tgt_np, pred_pose_np,
                title=f"Predicted Registration - Pair {idx}",
                save_path=f"registration_pair_{idx}_predicted.png"
            )
    
    # Show ground truth (identity) transformation
    visualize_point_cloud_registration(
        src_np, tgt_np, gt_pose_np,
        title=f"Ground Truth Registration - Pair {idx}",
        save_path=f"registration_pair_{idx}_gt.png"
    )


class PNGPointCloudDataset(Dataset):
    def __init__(self, png_paths, fx, fy, cx, cy, max_points=8192):
        self.png_paths = png_paths
        self.fx = fx
        self.fy = fy
        self.cx = cx  # Fixed: was missing cx assignment
        self.cy = cy
        self.max_points = max_points

    def __len__(self):
        return len(self.png_paths) - 1  # so we can pair i and i+1

    def __getitem__(self, idx):
        src = png_to_pointcloud(self.png_paths[idx], self.fx, self.fy, self.cx, self.cy, max_points=self.max_points)
        tgt = png_to_pointcloud(self.png_paths[idx+1], self.fx, self.fy, self.cx, self.cy, max_points=self.max_points)
        gt_pose = np.eye(4, dtype=np.float32)

        # Convert to torch tensors - Remove batch dimension since dataloader will add it
        src = torch.from_numpy(src).float()  # [N, 3]
        tgt = torch.from_numpy(tgt).float()  # [N, 3]
        gt_pose = torch.from_numpy(gt_pose).float()  # [4, 4]

        # For PNG data, we need to return in the format expected by synthetic data processing
        # Return as [N, 3] point clouds, not voxelized data
        return src, tgt, gt_pose
    
    def visualize_pair(self, idx, model=None, device='cpu'):
        """Convenience method to visualize a pair"""
        return visualize_dataset_pair(self, idx, model, device)