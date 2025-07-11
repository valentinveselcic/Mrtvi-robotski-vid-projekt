""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import argparse
import os
import logging
import torch
import torch.utils.data
import torchvision
import sys
import data_utils
import trainer

# LOGGER = logging.getLogger(__name__)
# LOGGER.addHandler(logging.NullHandler())

sys.path.append('.')  # Ensure current directory is in path
from rgrv_project_helper_functions import PNGPointCloudDataset  # Adjust import if needed

def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')
    
    # io settings.
    parser.add_argument('--outfile', type=str, default='./test_logs/2021_04_17_test_on_3dmatch_trained_on_modelnet',
                        metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/ThreeDMatch',
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('--categoryfile', type=str, default='./dataset/test_3dmatch.txt',
                        metavar='PATH', choices=['./dataset/test_3dmatch.txt', './dataset/modelnet40_half2.txt'], 
                        help='path to the categories to be tested')
    parser.add_argument('--pose_file', type=str, default='./dataset/gt_poses.csv',
                        metavar='PATH', help='path to the testing pose files')

    # settings for input data
    parser.add_argument('--dataset_type', default='3dmatch', type=str,
                        metavar='DATASET', choices=['modelnet', '3dmatch', 'png'], help='dataset type')
    parser.add_argument('--data_type', default='real', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--num_points', default=1000, type=int,
                        metavar='N', help='points in point-cloud')
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=0, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for voxelization
    parser.add_argument('--overlap_ratio', default=0.7, type=float,
                        metavar='D', help='overlapping ratio for 3DMatch dataset.')
    parser.add_argument('--voxel_ratio', default=0.05, type=float,
                        metavar='D', help='voxel ratio')
    parser.add_argument('--voxel', default=2, type=float,
                        metavar='D', help='how many voxels you want to divide in each axis')
    parser.add_argument('--max_voxel_points', default=1000, type=int,
                        metavar='N', help='maximum points allowed in a voxel')
    parser.add_argument('--num_voxels', default=8, type=int,
                        metavar='N', help='number of voxels')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='whether to visualize or not')
    parser.add_argument('--voxel_after_transf', action='store_true', default=False,
                        help='given voxelization before or after transformation')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='pointnet')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')

    # settings for LK
    parser.add_argument('-mi', '--max_iter', default=20, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    # settings for log
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--pretrained', default='./logs/model_trained_on_ModelNet40_model_best.pth', type=str,
                        metavar='PATH', help='path to pretrained model file ')
    
    args = parser.parse_args(argv)
    return args


def test(args, testset, dptnetlk):
    # Force CPU usage if CUDA is not available or not compiled with CUDA
    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        try:
            # Test if CUDA actually works
            torch.zeros(1).cuda()
        except (AssertionError, RuntimeError):
            print("CUDA not available or not compiled with CUDA support. Using CPU.")
            args.device = 'cpu'
    
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    model = dptnetlk.create_model()

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

    model.to(args.device)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # testing
    # LOGGER.debug('Begin Testing!')
    dptnetlk.test_one_epoch(model, testloader, args.device, 'test', args.data_type, args.vis)

    # Add visualization after testing
    print("\nGenerating visualizations...")
    testset.visualize_pair(4)  # Failed pair
    testset.visualize_pair(5)  # Good pair


def main(args):
    testset = get_datasets(args)
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    test(args, testset, dptnetlk)


def get_datasets(args):
    cinfo = None
    if args.categoryfile and args.data_type=='synthetic':
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    if args.dataset_type == 'png':
        # Set data_type to synthetic for PNG datasets to use the correct processing path
        args.data_type = 'synthetic'  # Add this line
        
        # Example: list of your PNG file paths
        png_paths = [
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/0.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/1.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/2.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/3.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/4.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/5.png',
            'miniBRANCH/tree_1_V_0001/Filtered__noGrass/depth/6.png'
        ]
        # Set intrinsics (replace with your actual values or estimation)
        fx = fy = 525.0
        cx = 640 / 2 - 0.5
        cy = 480 / 2 - 0.5
        testset = PNGPointCloudDataset(png_paths, fx, fy, cx, cy)

    elif args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube()])

        testdata = data_utils.ModelNet(args.dataset_path, train=-1, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration_fixed_perturbation(testdata, args.pose_file, sigma=args.sigma, clip=args.clip)
        
    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube()])

        testdata = data_utils.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration_fixed_perturbation(testdata, args.pose_file, sigma=args.sigma, clip=args.clip)

    elif args.dataset_type == '3dmatch':
        testset = data_utils.ThreeDMatch_Testing(args.dataset_path, args.categoryfile, args.overlap_ratio, 
                                                 args.voxel_ratio, args.voxel, args.max_voxel_points, 
                                                 args.num_voxels, args.pose_file, args.vis, args.voxel_after_transf)
    
    return testset

def visualize_dataset_pair(dataset, idx, model=None, device='cpu'):
    """
    Visualize a specific pair from the dataset with debug info
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
    
    # DEBUG: Check ground truth transformation
    print(f"Ground truth transformation:")
    print(gt_pose_np)
    
    # If model is provided, get predicted transformation
    if model is not None:
        model.eval()
        with torch.no_grad():
            # Add batch dimension
            src_batch = src.unsqueeze(0).to(device)
            tgt_batch = tgt.unsqueeze(0).to(device)
            
            # Get prediction - THIS IS THE KEY PART
            pred_pose = model(src_batch, tgt_batch)  # Make sure this calls the right method
            pred_pose_np = pred_pose.cpu().numpy()[0]
            
            # DEBUG: Check predicted transformation
            print(f"Predicted transformation:")
            print(pred_pose_np)
            
            # Show predicted transformation
            visualize_point_cloud_registration(
                src_np, tgt_np, pred_pose_np,
                title=f"Predicted Registration - Pair {idx}",
                save_path=f"registration_pair_{idx}_predicted.png"
            )
    else:
        print("No model provided - skipping predicted transformation")
    
    # Show ground truth (identity) transformation
    visualize_point_cloud_registration(
        src_np, tgt_np, gt_pose_np,
        title=f"Ground Truth Registration - Pair {idx}",
        save_path=f"registration_pair_{idx}_gt.png"
    )

    
if __name__ == '__main__':
    ARGS = options()

    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
    #     filename=ARGS.logfile)
    # LOGGER.debug('Testing (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)

    # LOGGER.debug('Testing completed! Hahaha~~ (PID=%d)', os.getpid())
