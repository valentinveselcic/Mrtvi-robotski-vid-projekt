#!/usr/bin/env python3
"""
Clean inference test for PointNetLK on PNG depth images
Addresses gradient computation issues and proper model calling
"""
import os  # Add this import at the top of the file
import logging
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# Suppress debug messages
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
Image.MAX_IMAGE_PIXELS = None

# Import project modules
import model
import trainer
from rgrv_project_helper_functions import (
    PNGPointCloudDataset, 
    visualize_point_cloud_registration,
    apply_transformation, png_to_pointcloud
)

def setup_model(device='cpu'):
    """Setup the trained PointNetLK model"""
    # Create trainer and model
    class Args:
        def __init__(self):
            self.dim_k = 1024
            self.device = device
            self.max_iter = 10
            self.embedding = 'pointnet'
            self.outfile = 'test_results'
    
    args = Args()
    ptnetlk_trainer = trainer.TrainerAnalyticalPointNetLK(args)
    ptnetlk = ptnetlk_trainer.create_model()
    
    # Load trained weights
    model_path = 'logs/model_trained_on_ModelNet40_model_best.pth'
    try:
        checkpoint = torch.load(model_path, map_location=device)
        ptnetlk.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    ptnetlk.to(device)
    ptnetlk.eval()
    return ptnetlk

def test_single_pair_inference(ptnetlk, src_points, tgt_points, device='cpu', max_iter=20):
    """
    Test inference on a single point cloud pair with proper gradient setup
    """
    # Ensure tensors have the right properties for gradient computation
    src_tensor = torch.from_numpy(src_points).float().unsqueeze(0).to(device)  # [1, N, 3]
    tgt_tensor = torch.from_numpy(tgt_points).float().unsqueeze(0).to(device)  # [1, N, 3]
    
    # CRITICAL: Enable gradients for proper jacobian computation
    src_tensor.requires_grad_(True)
    tgt_tensor.requires_grad_(True)
    
    print(f"Source shape: {src_tensor.shape}, Target shape: {tgt_tensor.shape}")
    print(f"Source requires_grad: {src_tensor.requires_grad}, Target requires_grad: {tgt_tensor.requires_grad}")
    
    try:
        # Use the correct static method for inference
        with torch.enable_grad():  # Ensure gradients are enabled
            _ = model.AnalyticalPointNetLK.do_forward(
                ptnetlk, 
                src_tensor, None,  # p0, voxel_coords_p0 (None for synthetic data)
                tgt_tensor, None,  # p1, voxel_coords_p1 (None for synthetic data)
                maxiter=max_iter,
                xtol=1.0e-7,
                p0_zero_mean=True,
                p1_zero_mean=True,
                mode='test',
                data_type='synthetic',  # Use synthetic for PNG data
                num_random_points=100
            )
        
        # Get the estimated transformation
        estimated_transform = ptnetlk.g.detach().cpu().numpy()[0]  # [4, 4]
        iterations = ptnetlk.itr
        
        print(f"✓ Inference successful! Iterations: {iterations}")
        print(f"Estimated transformation:")
        print(estimated_transform)
        
        return estimated_transform, iterations
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def test_png_registration():
    """
    Test point cloud registration on PNG depth images
    """
    print("=" * 60)
    print("CLEAN PNG POINT CLOUD REGISTRATION TEST")
    print("=" * 60)
    
    # Setup
    device = 'cpu'  # Use CPU to avoid CUDA issues
    print(f"Using device: {device}")
    
    # Load model
    ptnetlk = setup_model(device)
    if ptnetlk is None:
        print("Failed to load model, exiting...")
        return
    
    # Create dataset
    png_paths = [
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/0.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/1.png',
        'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/2.png',
    ]
    
    # Camera intrinsics
    fx = fy = 525.0
    cx = 640 / 2 - 0.5
    cy = 480 / 2 - 0.5
    
    # Create dataset with Z-filtering to remove background
    dataset = PNGPointCloudDataset(png_paths, fx, fy, cx, cy)
    
    print(f"Dataset created with {len(dataset)} pairs")
    
    # Test inference on first pair
    if len(dataset) > 0:
        src, tgt, gt_pose = dataset[1]
        
        # Convert to numpy for processing
        src_np = src.numpy()
        tgt_np = tgt.numpy()
        gt_pose_np = gt_pose.numpy()
        
        print(f"\nTesting pair 0:")
        print(f"Source points: {src_np.shape[0]}")
        print(f"Target points: {tgt_np.shape[0]}")
        print(f"Ground truth pose:")
        print(gt_pose_np)
        
        # Run inference
        print("\nRunning inference...")
        pred_transform, iterations = test_single_pair_inference(
            ptnetlk, src_np, tgt_np, device, max_iter=10
        )
        
        if pred_transform is not None:
            print(f"\n✓ Registration successful!")
            print(f"Iterations: {iterations}")
            print(f"Predicted transformation:")
            print(pred_transform)
            
            # Compute registration error
            error_matrix = pred_transform @ np.linalg.inv(gt_pose_np)
            translation_error = np.linalg.norm(error_matrix[:3, 3])
            rotation_error = np.arccos(np.clip((np.trace(error_matrix[:3, :3]) - 1) / 2, -1, 1))
            
            print(f"\nRegistration Metrics:")
            print(f"Translation error: {translation_error:.6f}")
            print(f"Rotation error: {rotation_error:.6f} radians ({np.degrees(rotation_error):.2f} degrees)")
            
            # Visualize results
            try:
                print("\nGenerating visualization...")
                visualize_point_cloud_registration(
                    src_np, tgt_np, pred_transform,
                    title=f"PointNetLK Registration - Pair 0 ({iterations} iterations)",
                    save_path="clean_inference_result.png"
                )
                print("✓ Visualization saved as 'clean_inference_result.png'")
            except Exception as e:
                print(f"✗ Visualization failed: {e}")
        else:
            print("\n✗ Registration failed")
    
    # Test other consecutive pairs to see if the pattern is consistent
    for i in range(min(3, len(dataset))):
        print(f"\n=== Testing Pair {i} ===")
        src, tgt, gt_pose = dataset[i]
        
        # Convert to numpy for processing
        src_np = src.numpy()
        tgt_np = tgt.numpy()
        gt_pose_np = gt_pose.numpy()
        
        print(f"Source points: {src_np.shape[0]}")
        print(f"Target points: {tgt_np.shape[0]}")
        print(f"Ground truth pose:")
        print(gt_pose_np)
        
        # Run inference
        pred_transform, iterations = test_single_pair_inference(
            ptnetlk, src_np, tgt_np, device, max_iter=10
        )
        
        if pred_transform is not None:
            print(f"✓ Registration successful! Iterations: {iterations}")
            print(f"Predicted transformation:")
            print(pred_transform)
        else:
            print("✗ Registration failed")
    
    print("\n" + "=" * 60)
    print("Test completed!")

def comprehensive_reliability_test(idx = 1):
    """
    Test model reliability across all available pairs
    """
    print("=" * 60)
    print("COMPREHENSIVE RELIABILITY TEST")
    print("=" * 60)
    
    # Setup
    device = 'cpu'
    ptnetlk = setup_model(device)
    if ptnetlk is None:
        return
    
    # Load all available PNG files
    if idx == 1:
        png_paths = [
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/2.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/3.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/5.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/6.png',
        ]
    elif idx == 9:
        png_paths = [
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/3.png',
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/5.png',
        ]
    elif idx == 11:
        png_paths = [
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/5.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/6.png',
        ]
    elif idx == 12:
        png_paths = [
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/5.png',
        ]
    
    fx = fy = 525.0
    cx = cy = 320.0
    dataset = PNGPointCloudDataset(png_paths, fx, fy, cx, cy)
    
    # Reliability metrics
    results = {
        'successes': 0,
        'failures': 0,
        'translation_errors': [],
        'rotation_errors': [],
        'iterations': [],
        'convergence_rate': [],
        'pair_results': []
    }
    
    print(f"Testing {len(dataset)} pairs...")
    
    # Test all pairs
    for i in range(len(dataset)):
        print(f"\n--- Pair {i}: {png_paths[i]} → {png_paths[i+1]} ---")
        
        src, tgt, gt_pose = dataset[i]
        src_np = src.numpy()
        tgt_np = tgt.numpy()
        gt_pose_np = gt_pose.numpy()
        
        # Test with multiple iteration limits to check robustness
        for max_iter in [20]:
            print(f"  Testing with max_iter={max_iter}...", end=" ")
            
            pred_transform, iterations = test_single_pair_inference(
                ptnetlk, src_np, tgt_np, device, max_iter=max_iter
            )
            
            if pred_transform is not None:
                # Compute errors
                error_matrix = pred_transform @ np.linalg.inv(gt_pose_np)
                trans_error = np.linalg.norm(error_matrix[:3, 3])
                rot_error = np.arccos(np.clip((np.trace(error_matrix[:3, :3]) - 1) / 2, -1, 1))
                
                # Determine if "successful" (reasonable errors)
                is_success = trans_error < 0.1 and rot_error < np.pi/2  # 10cm, 90deg thresholds
                
                if is_success:
                    results['successes'] += 1
                    print("✓ SUCCESS")
                else:
                    results['failures'] += 1
                    print("✗ HIGH ERROR")
                
                results['translation_errors'].append(trans_error)
                results['rotation_errors'].append(np.degrees(rot_error))
                results['iterations'].append(iterations)
                results['convergence_rate'].append(iterations / max_iter)
                
                results['pair_results'].append({
                    'pair_idx': i,
                    'max_iter': max_iter,
                    'iterations': iterations,
                    'trans_error': trans_error,
                    'rot_error_deg': np.degrees(rot_error),
                    'success': is_success,
                    'converged': iterations < max_iter
                })
                
            else:
                results['failures'] += 1
                print("✗ FAILED")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RELIABILITY ANALYSIS")
    print("=" * 60)
    
    total_tests = results['successes'] + results['failures']
    success_rate = results['successes'] / total_tests * 100 if total_tests > 0 else 0
    
    print(f"Success Rate: {success_rate:.1f}% ({results['successes']}/{total_tests})")
    
    if results['translation_errors']:
        print(f"Translation Error - Mean: {np.mean(results['translation_errors']):.4f}m")
        print(f"Translation Error - Std:  {np.std(results['translation_errors']):.4f}m")
        print(f"Translation Error - Max:  {np.max(results['translation_errors']):.4f}m")
        
        print(f"Rotation Error - Mean: {np.mean(results['rotation_errors']):.1f}°")
        print(f"Rotation Error - Std:  {np.std(results['rotation_errors']):.1f}°")
        print(f"Rotation Error - Max:  {np.max(results['rotation_errors']):.1f}°")
        
        print(f"Convergence - Mean iterations: {np.mean(results['iterations']):.1f}")
        print(f"Convergence - Mean rate: {np.mean(results['convergence_rate']):.1f}")
    
    # Identify problematic pairs
    print(f"\nProblematic pairs (high error or non-convergence):")
    for result in results['pair_results']:
        if not result['success'] or not result['converged']:
            print(f"  Pair {result['pair_idx']}: Trans={result['trans_error']:.3f}m, "
                  f"Rot={result['rot_error_deg']:.1f}°, Iter={result['iterations']}/{result['max_iter']}")
    
    return results

def comprehensive_reliability_test_with_visualization():
    """
    Test all trees with iterative visualization, saving only first and last iterations
    organized by tree folders
    """
    print("=" * 60)
    print("COMPREHENSIVE RELIABILITY TEST WITH SELECTIVE VISUALIZATION")
    print("=" * 60)
    
    # Setup
    device = 'cpu'
    ptnetlk = setup_model(device)
    if ptnetlk is None:
        return
    # Skip tree_1_v_0001 i 0009 (vec imam za njega sve)
    # Define all tree datasets
    tree_datasets = {
        # 'tree_1_V_0001': [
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/0.png',
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/1.png',
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/2.png',
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/3.png',
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/4.png',
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/5.png',
        #     'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/6.png',
        # ],
        # 'tree_1_V_0009': [
        #     'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/0.png',
        #     'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/3.png',
        #     'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/4.png',
        #     'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/5.png',
        # ],
        'tree_1_V_0011': [
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/5.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/6.png',
        ],
        'tree_1_V_0012': [
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/5.png',
        ]
    }
    
    fx = fy = 525.0
    cx = cy = 320.0
    
    # Overall results tracking
    overall_results = {
        'tree_results': {},
        'total_successes': 0,
        'total_failures': 0,
        'all_translation_errors': [],
        'all_rotation_errors': [],
        'all_iterations': []
    }
    
    # Process each tree
    for tree_name, png_paths in tree_datasets.items():
        print(f"\n{'='*40}")
        print(f"PROCESSING {tree_name.upper()}")
        print(f"{'='*40}")
        
        # Create tree-specific output directory
        tree_output_dir = f"results/{tree_name}"
        os.makedirs(tree_output_dir, exist_ok=True)
        
        # Create dataset for this tree
        dataset = PNGPointCloudDataset(png_paths, fx, fy, cx, cy)
        
        # Tree-specific results
        tree_results = {
            'successes': 0,
            'failures': 0,
            'translation_errors': [],
            'rotation_errors': [],
            'iterations': [],
            'pair_results': []
        }
        
        print(f"Testing {len(dataset)} pairs from {tree_name}...")
        
        # Test all pairs in this tree
        for pair_idx in range(len(dataset)):
            pair_name = f"pair_{pair_idx}_{os.path.basename(png_paths[pair_idx]).split('.')[0]}_to_{os.path.basename(png_paths[pair_idx+1]).split('.')[0]}"
            print(f"\n--- {pair_name} ---")
            
            src, tgt, gt_pose = dataset[pair_idx]
            src_np = src.numpy()
            tgt_np = tgt.numpy()
            gt_pose_np = gt_pose.numpy()
            
            print(f"Source: {os.path.basename(png_paths[pair_idx])}")
            print(f"Target: {os.path.basename(png_paths[pair_idx+1])}")
            print(f"Source points: {src_np.shape[0]}, Target points: {tgt_np.shape[0]}")
            
            # Test with iterative visualization (first and last only)
            final_transform, iterations, all_transforms = test_with_selective_visualization(
                ptnetlk, src_np, tgt_np, device, 
                max_iter=20, 
                output_dir=tree_output_dir,
                pair_name=pair_name
            )
            
            if final_transform is not None:
                # Compute errors
                error_matrix = final_transform @ np.linalg.inv(gt_pose_np)
                trans_error = np.linalg.norm(error_matrix[:3, 3])
                rot_error = np.arccos(np.clip((np.trace(error_matrix[:3, :3]) - 1) / 2, -1, 1))
                
                # Determine success
                is_success = trans_error < 0.1 and rot_error < np.pi/2  # 10cm, 90deg thresholds
                
                if is_success:
                    tree_results['successes'] += 1
                    overall_results['total_successes'] += 1
                    print(">> SUCCESS")
                else:
                    tree_results['failures'] += 1
                    overall_results['total_failures'] += 1
                    print(">> HIGH ERROR")
                
                # Store results
                tree_results['translation_errors'].append(trans_error)
                tree_results['rotation_errors'].append(np.degrees(rot_error))
                tree_results['iterations'].append(iterations)
                
                overall_results['all_translation_errors'].append(trans_error)
                overall_results['all_rotation_errors'].append(np.degrees(rot_error))
                overall_results['all_iterations'].append(iterations)
                
                tree_results['pair_results'].append({
                    'pair_name': pair_name,
                    'pair_idx': pair_idx,
                    'iterations': iterations,
                    'trans_error': trans_error,
                    'rot_error_deg': np.degrees(rot_error),
                    'success': is_success,
                    'converged': iterations < 20
                })
                
                print(f"  Trans error: {trans_error:.4f}m, Rot error: {np.degrees(rot_error):.1f}°, Iterations: {iterations}")
                
            else:
                tree_results['failures'] += 1
                overall_results['total_failures'] += 1
                print(">> REGISTRATION FAILED")
        
        # Save tree-specific results
        save_tree_results(tree_results, tree_name, tree_output_dir)
        overall_results['tree_results'][tree_name] = tree_results
    
    # Print overall summary
    print_overall_summary(overall_results)
    
    return overall_results

def test_with_selective_visualization(ptnetlk, src_points, tgt_points, device='cpu', 
                                    max_iter=20, output_dir="results", pair_name="pair"):
    """
    Test inference with visualization of only first and last iterations
    """
    # Setup tensors
    src_tensor = torch.from_numpy(src_points).float().unsqueeze(0).to(device)
    tgt_tensor = torch.from_numpy(tgt_points).float().unsqueeze(0).to(device)
    src_tensor.requires_grad_(True)
    tgt_tensor.requires_grad_(True)
    
    print(f"  Running inference (max {max_iter} iterations)...", end=" ")
    
    # Store transformations
    transformations = []
    
    try:
        with torch.enable_grad():
            # Run full inference to get all intermediate results
            _ = model.AnalyticalPointNetLK.do_forward(
                ptnetlk, 
                src_tensor, None,
                tgt_tensor, None,
                maxiter=max_iter,
                xtol=1.0e-7,
                p0_zero_mean=True,
                p1_zero_mean=True,
                mode='test',
                data_type='synthetic',
                num_random_points=100
            )
            
            # Get final transformation and iteration count
            final_transform = ptnetlk.g.detach().cpu().numpy()[0]
            iterations = ptnetlk.itr
            
            print(f"✓ ({iterations} iterations)")
            
            # Create visualizations for first and last iterations only
            
            # First iteration (identity matrix)
            identity_transform = np.eye(4)
            visualize_point_cloud_registration(
                src_points, tgt_points, identity_transform,
                title=f"{pair_name} - Initial State (Iteration 0)",
                save_path=f"{output_dir}/{pair_name}_iteration_00_initial.png",
                subsample=1000
            )
            
            # Last iteration (final result)
            visualize_point_cloud_registration(
                src_points, tgt_points, final_transform,
                title=f"{pair_name} - Final Result (Iteration {iterations})",
                save_path=f"{output_dir}/{pair_name}_iteration_{iterations:02d}_final.png",
                subsample=1000
            )
            
            print(f"  ✓ Saved initial and final visualizations to {output_dir}/")
            
            return final_transform, iterations, [identity_transform, final_transform]
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None, 0, []

def save_tree_results(tree_results, tree_name, output_dir):
    """
    Save detailed results for a specific tree
    """
    import json
    from datetime import datetime
    
    # Save to text file with UTF-8 encoding
    txt_path = f"{output_dir}/{tree_name}_results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"POINTNETLK REGISTRATION RESULTS FOR {tree_name.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_tests = tree_results['successes'] + tree_results['failures']
        success_rate = tree_results['successes'] / total_tests * 100 if total_tests > 0 else 0
        
        f.write(f"SUMMARY\n")
        f.write(f"Success Rate: {success_rate:.1f}% ({tree_results['successes']}/{total_tests})\n")
        
        if tree_results['translation_errors']:
            f.write(f"Translation Error - Mean: {np.mean(tree_results['translation_errors']):.4f}m\n")
            f.write(f"Translation Error - Std:  {np.std(tree_results['translation_errors']):.4f}m\n")
            f.write(f"Translation Error - Max:  {np.max(tree_results['translation_errors']):.4f}m\n")
            
            f.write(f"Rotation Error - Mean: {np.mean(tree_results['rotation_errors']):.1f}°\n")
            f.write(f"Rotation Error - Std:  {np.std(tree_results['rotation_errors']):.1f}°\n")
            f.write(f"Rotation Error - Max:  {np.max(tree_results['rotation_errors']):.1f}°\n")
            
            f.write(f"Convergence - Mean iterations: {np.mean(tree_results['iterations']):.1f}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 40 + "\n")
        for result in tree_results['pair_results']:
            f.write(f"{result['pair_name']}:\n")
            f.write(f"  Success: {'YES' if result['success'] else 'NO'}\n")
            f.write(f"  Translation error: {result['trans_error']:.4f}m\n")
            f.write(f"  Rotation error: {result['rot_error_deg']:.1f}°\n")
            f.write(f"  Iterations: {result['iterations']}\n")
            f.write(f"  Converged: {'YES' if result['converged'] else 'NO'}\n\n")
    
    # Create a copy of tree_results with converted types for JSON
    json_results = {
        'successes': int(tree_results['successes']),
        'failures': int(tree_results['failures']),
        'translation_errors': [float(x) for x in tree_results['translation_errors']],
        'rotation_errors': [float(x) for x in tree_results['rotation_errors']],
        'iterations': [int(x) for x in tree_results['iterations']],
        'pair_results': []
    }
    
    # Convert pair results
    for result in tree_results['pair_results']:
        json_results['pair_results'].append({
            'pair_name': str(result['pair_name']),
            'pair_idx': int(result['pair_idx']),
            'iterations': int(result['iterations']),
            'trans_error': float(result['trans_error']),
            'rot_error_deg': float(result['rot_error_deg']),
            'success': bool(result['success']),
            'converged': bool(result['converged'])
        })
    
    # Save to JSON file
    json_path = f"{output_dir}/{tree_name}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"  >> Tree results saved to {txt_path} and {json_path}")

def print_overall_summary(overall_results):
    """
    Print comprehensive summary across all trees
    """
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY ACROSS ALL TREES")
    print("=" * 80)
    
    total_tests = overall_results['total_successes'] + overall_results['total_failures']
    overall_success_rate = overall_results['total_successes'] / total_tests * 100 if total_tests > 0 else 0
    
    print(f"Overall Success Rate: {overall_success_rate:.1f}% ({overall_results['total_successes']}/{total_tests})")
    
    # Per-tree breakdown
    print(f"\nPer-tree breakdown:")
    for tree_name, tree_data in overall_results['tree_results'].items():
        tree_total = tree_data['successes'] + tree_data['failures']
        tree_rate = tree_data['successes'] / tree_total * 100 if tree_total > 0 else 0
        print(f"  {tree_name}: {tree_rate:.1f}% ({tree_data['successes']}/{tree_total})")
    
    # Overall statistics
    if overall_results['all_translation_errors']:
        print(f"\nOverall Statistics:")
        print(f"Translation Error - Mean: {np.mean(overall_results['all_translation_errors']):.4f}m")
        print(f"Translation Error - Std:  {np.std(overall_results['all_translation_errors']):.4f}m")
        print(f"Rotation Error - Mean: {np.mean(overall_results['all_rotation_errors']):.1f}°")
        print(f"Rotation Error - Std:  {np.std(overall_results['all_rotation_errors']):.1f}°")
        print(f"Convergence - Mean iterations: {np.mean(overall_results['all_iterations']):.1f}")
    
    print(f"\n>> All results saved in organized tree folders under 'results/'")
    print("=" * 80)
    
def reconstruct_complete_tree_models():
    """
    Reconstruct complete 3D models for all trees using multi-view registration
    """
    print("=" * 60)
    print("COMPLETE TREE MODEL RECONSTRUCTION")
    print("=" * 60)
    
    # Setup
    device = 'cpu'
    ptnetlk = setup_model(device)
    if ptnetlk is None:
        return
    
    # Define tree datasets
    tree_datasets = {
        'tree_1_V_0001': [
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/2.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/3.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/5.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/6.png',
        ],
        'tree_1_V_0009': [
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/3.png',
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0009/Filtered__noGrass/depth/5.png',
        ],
        'tree_1_V_0011': [
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/5.png',
            'miniBRANCH/tree_1_V_0011/Filtered__noGrass/depth/6.png',
        ],
        'tree_1_V_0012': [
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0012/Filtered__noGrass/depth/5.png',
        ]
    }
    
    fx = fy = 525.0
    cx = cy = 320.0
    
    # Reconstruct each tree
    for tree_name, png_paths in tree_datasets.items():
        print(f"\n🌳 RECONSTRUCTING {tree_name.upper()}")
        print("=" * 50)
        
        # Create output directory
        output_dir = f"results/{tree_name}/reconstruction"
        os.makedirs(output_dir, exist_ok=True)
        
        # Reconstruct using sequential registration
        complete_model, poses = reconstruct_tree_sequential(
            tree_name, png_paths, fx, fy, cx, cy, ptnetlk, device
        )
        
        # Save complete model
        save_tree_reconstruction(complete_model, poses, tree_name, output_dir)
        
        # Visualize complete model
        visualize_complete_tree_model(complete_model, tree_name, output_dir)
    
    print("\n🎉 All tree reconstructions completed!")

def reconstruct_tree_sequential(tree_name, png_paths, fx, fy, cx, cy, ptnetlk, device):
    """
    Reconstruct tree using sequential registration
    """
    print(f"Loading {len(png_paths)} depth images...")
    
    # Load all point clouds
    point_clouds = []
    for i, png_path in enumerate(png_paths):
        pc = png_to_pointcloud(png_path, fx, fy, cx, cy, max_points=4096)
        point_clouds.append({
            'points': pc,
            'frame_id': i,
            'path': png_path,
            'pose': np.eye(4)
        })
        print(f"  Frame {i}: {pc.shape[0]} points from {os.path.basename(png_path)}")
    
    # Sequential registration
    print("\nPerforming sequential registration...")
    accumulated_poses = [np.eye(4)]
    
    for i in range(1, len(point_clouds)):
        print(f"  Registering frame {i} to accumulated model...", end=" ")
        
        # Register to previous frame
        src = point_clouds[i]['points']
        tgt = point_clouds[i-1]['points']
        
        rel_transform, iterations = test_single_pair_inference(
            ptnetlk, src, tgt, device, max_iter=20
        )
        
        if rel_transform is not None:
            # Accumulate transformation
            global_pose = accumulated_poses[-1] @ rel_transform
            accumulated_poses.append(global_pose)
            point_clouds[i]['pose'] = global_pose
            
            # Compute registration error for quality assessment
            error = np.linalg.norm(rel_transform[:3, 3])
            print(f"✓ ({iterations} iter, {error:.3f}m translation)")
        else:
            print("✗ Failed, using identity")
            accumulated_poses.append(accumulated_poses[-1])
    
    # Transform all point clouds to global coordinates
    print("\nCombining all point clouds...")
    global_point_clouds = []
    
    for i, pc_data in enumerate(point_clouds):
        transformed_points = apply_transformation(pc_data['points'], pc_data['pose'])
        global_point_clouds.append(transformed_points)
        print(f"  Frame {i}: transformed {transformed_points.shape[0]} points")
    
    # Combine into single model
    complete_model = np.vstack(global_point_clouds)
    print(f"\n✓ Complete model: {complete_model.shape[0]} points")
    
    return complete_model, accumulated_poses

def save_tree_reconstruction(complete_model, poses, tree_name, output_dir):
    """
    Save the complete tree reconstruction
    """
    # Save point cloud as PLY file
    ply_path = f"{output_dir}/{tree_name}_complete_model.ply"
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(complete_model)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in complete_model:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    # Save poses
    poses_path = f"{output_dir}/{tree_name}_camera_poses.npy"
    np.save(poses_path, np.array(poses))
    
    print(f"  ✓ Saved complete model: {ply_path}")
    print(f"  ✓ Saved camera poses: {poses_path}")

def visualize_complete_tree_model(complete_model, tree_name, output_dir):
    """
    Create visualizations of the complete tree model
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(complete_model[:, 0], complete_model[:, 1], complete_model[:, 2], 
               c=complete_model[:, 2], s=0.5, alpha=0.6, cmap='viridis')
    ax1.set_title(f'{tree_name} - Complete 3D Model')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # Top view (X-Y plane)
    ax2 = fig.add_subplot(132)
    ax2.scatter(complete_model[:, 0], complete_model[:, 1], 
               c=complete_model[:, 2], s=0.5, alpha=0.6, cmap='viridis')
    ax2.set_title('Top View (X-Y)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.axis('equal')
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(133)
    ax3.scatter(complete_model[:, 0], complete_model[:, 2], 
               c=complete_model[:, 1], s=0.5, alpha=0.6, cmap='plasma')
    ax3.set_title('Side View (X-Z)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.axis('equal')
    
    plt.tight_layout()
    
    viz_path = f"{output_dir}/{tree_name}_complete_model_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  ✓ Saved visualization: {viz_path}")

if __name__ == "__main__":
    # test_png_registration()
    # comprehensive_reliability_test(12)
    # comprehensive_reliability_test_with_visualization()
    reconstruct_complete_tree_models()


def main():
    pass

main()