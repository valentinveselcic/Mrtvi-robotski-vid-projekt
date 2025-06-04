import torch
import os
import json
import logging
import numpy as np
from datetime import datetime
from rgrv_project_helper_functions import PNGPointCloudDataset, visualize_dataset_pair
from model import Model
import glob


def save_metrics_to_file(results, save_dir="results"):
    """Save metrics to text and JSON files"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to text file (human readable)
    txt_path = os.path.join(save_dir, f"visualization_results_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("POINTNETLK VISUALIZATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Visualization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total pairs visualized: {len(results['visualized_pairs'])}\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"PNG files: {len(results['png_files'])}\n")
        for i, file in enumerate(results['png_files']):
            f.write(f"  {i}: {file}\n")
        f.write(f"\nCamera parameters:\n")
        f.write(f"  fx, fy: {results['camera_params']['fx']}, {results['camera_params']['fy']}\n")
        f.write(f"  cx, cy: {results['camera_params']['cx']}, {results['camera_params']['cy']}\n")
        f.write(f"  max_points: {results['camera_params']['max_points']}\n")
        
        # Visualization details
        f.write("\nVISUALIZED PAIRS\n")
        f.write("-" * 30 + "\n")
        for pair_info in results['visualized_pairs']:
            f.write(f"\nPair {pair_info['idx']}: {pair_info['src_file']} → {pair_info['tgt_file']}\n")
            f.write(f"  Source points: {pair_info['src_points']}\n")
            f.write(f"  Target points: {pair_info['tgt_points']}\n")
            f.write(f"  Model loaded: {pair_info['model_loaded']}\n")
            f.write(f"  Files saved: {pair_info['files_saved']}\n")
    
    # Save to JSON file (machine readable)
    json_path = os.path.join(save_dir, f"visualization_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  Text file: {txt_path}")
    print(f"  JSON file: {json_path}")
    
    return txt_path, json_path

def main():
    # Set up dataset
    png_paths = [
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/0.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/1.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/2.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/3.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/4.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/5.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/6.png'
    ]
    
    fx = fy = 525.0
    cx = cy = 320.0
    dataset = PNGPointCloudDataset(png_paths, fx, fy, cx, cy, max_points=2048)
    
    # Initialize results tracking
    results = {
        'png_files': png_paths,
        'camera_params': {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'max_points': 2048
        },
        'visualized_pairs': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Load trained model (optional)
    model = None
    device = 'cpu'
    model_loaded = False
    
    # Try to find and load model
    checkpoint_paths = glob.glob('logs/*.pth') + glob.glob('*.pth')
    if checkpoint_paths:
        try:
            model = Model()
            checkpoint = torch.load(checkpoint_paths[0], map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            print(f"Loaded model from: {checkpoint_paths[0]}")
            model_loaded = True
        except Exception as e:
            print(f"Could not load model: {e}")
            model = None
    
    # Visualize different pairs
    print("Generating visualizations...")
    
    pairs_to_visualize = [8, 9, 10]  # Failed, good, good
    
    for pair_idx in pairs_to_visualize:
        if pair_idx < len(dataset):
            print(f"\n=== Pair {pair_idx} ===")
            
            # Get point cloud info before visualization
            src, tgt, _ = dataset[pair_idx]
            src_points = len(src.numpy())
            tgt_points = len(tgt.numpy())
            
            # Expected output files
            expected_files = []
            if model_loaded:
                expected_files.append(f"registration_pair_{pair_idx}_predicted.png")
            expected_files.append(f"registration_pair_{pair_idx}_gt.png")
            
            # Run visualization
            dataset.visualize_pair(pair_idx, model, device)
            
            # Record results
            pair_result = {
                'idx': pair_idx,
                'src_file': os.path.basename(png_paths[pair_idx]),
                'tgt_file': os.path.basename(png_paths[pair_idx + 1]),
                'src_points': src_points,
                'tgt_points': tgt_points,
                'model_loaded': model_loaded,
                'files_saved': expected_files
            }
            results['visualized_pairs'].append(pair_result)
    
    # Save results summary
    save_metrics_to_file(results)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETED!")
    print(f"{'='*60}")
    print(f"Pairs visualized: {len(pairs_to_visualize)}")
    print(f"Files generated: {len(pairs_to_visualize) * (2 if model_loaded else 1)}")
    print("Check the 'results' folder for detailed logs.")

if __name__ == "__main__":
    main()