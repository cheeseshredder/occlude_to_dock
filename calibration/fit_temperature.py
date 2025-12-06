"""
Temperature Scaling Calibration: Fit temperature parameter for detector calibration.

This script loads the calibration dataset and finds the optimal temperature
that minimizes Expected Calibration Error (ECE).
"""

import numpy as np
import argparse
import yaml
from datetime import datetime
import sys
sys.path.append('..')

from calibration.collect_dataset import load_calibration_dataset
from perception.vision_model import PerceptionModule, compute_ECE, match_to_ground_truth
from utils import load_config, sigmoid, logit_inverse, create_camera_intrinsics, create_camera_extrinsics


def fit_temperature(calibration_data, perception_module, camera_params):
    """
    Fit temperature parameter using calibration dataset.
    
    Args:
        calibration_data: List of calibration samples
        perception_module: PerceptionModule (with T=1.0 initially)
        camera_params: Camera parameters
        
    Returns:
        best_T: Optimal temperature scalar
        metrics: Dict with ECE before/after calibration
    """
    print("\nRunning detections on calibration data...")
    
    # Collect raw logits and match to ground truth
    logits = []
    labels = []  # Binary: 1 if correct, 0 if incorrect
    
    for idx, sample in enumerate(calibration_data):
        if (idx + 1) % 20 == 0:
            print(f"  Processing sample {idx+1}/{len(calibration_data)}...")
        
        rgb = sample['rgb']
        depth = sample['depth']
        robot_pose = sample['robot_pose']
        gt = sample['ground_truth']
        
        # Run detection
        detections = perception_module.detect(
            rgb, depth, robot_pose, camera_params
        )
        
        # Match charger detections to ground truth
        for det in detections:
            if det['class'] == 'charger':
                is_correct = match_to_ground_truth(
                    det['bbox_2d'],
                    gt['charger_bbox'],
                    iou_threshold=0.5
                )
                
                logits.append(det['raw_logit'])
                labels.append(1 if is_correct else 0)
    
    print(f"\n✓ Collected {len(logits)} charger detections")
    
    if len(logits) == 0:
        print("ERROR: No charger detections found!")
        return 1.0, {'ECE_before': None, 'ECE_after': None}
    
    logits = np.array(logits)
    labels = np.array(labels)
    
    # Compute ECE before calibration (T=1.0)
    probs_before = sigmoid(logits)
    ECE_before = compute_ECE(probs_before, labels)
    
    print(f"\nECE before calibration (T=1.0): {ECE_before:.4f}")
    
    # Grid search for best T
    print("\nSearching for optimal temperature...")
    
    best_T = 1.0
    best_ECE = ECE_before
    
    T_values = np.linspace(0.5, 3.0, 26)  # 26 values from 0.5 to 3.0
    
    for T in T_values:
        probs_calibrated = sigmoid(logits / T)
        ECE = compute_ECE(probs_calibrated, labels)
        
        if ECE < best_ECE:
            best_ECE = ECE
            best_T = T
    
    print(f"\n✓ Optimal temperature: T = {best_T:.3f}")
    print(f"✓ ECE after calibration: {best_ECE:.4f}")
    print(f"✓ Improvement: {(ECE_before - best_ECE)/ECE_before*100:.1f}%")
    
    return best_T, {
        'ECE_before': float(ECE_before),
        'ECE_after': float(best_ECE),
        'improvement_percent': float((ECE_before - best_ECE)/ECE_before*100),
        'num_samples': int(len(logits))
    }


def save_calibration_config(T, metrics, output_path):
    """Save calibration results to YAML config."""
    
    config = {
        'temperature': float(T),
        'calibration_metrics': {
            'ECE_before': metrics['ECE_before'],
            'ECE_after': metrics['ECE_after'],
            'improvement_percent': metrics['improvement_percent'],
            'num_samples': metrics['num_samples'],
            'date_calibrated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'dataset': {
            'num_scenes': 10,
            'samples_per_scene': 20,
            'total_samples': 200,
            'dataset_path': 'calibration_dataset.pkl'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ Calibration config saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fit temperature for calibration')
    parser.add_argument('--dataset', type=str, 
                       default='calibration/calibration_dataset.pkl',
                       help='Path to calibration dataset')
    parser.add_argument('--output', type=str, 
                       default='config/calibration.yaml',
                       help='Output calibration config path')
    parser.add_argument('--config_dir', type=str, default='config',
                       help='Config directory for params')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Temperature Scaling Calibration")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load calibration dataset
    print("Loading calibration dataset...")
    calibration_data = load_calibration_dataset(args.dataset)
    
    # Load parameters
    params = load_config(os.path.join(args.config_dir, 'params.yaml'))
    
    # Initialize perception module (T=1.0 initially)
    print("\nInitializing perception module...")
    perception_config = params['perception']
    perception = PerceptionModule(
        model_name=perception_config['model_name'],
        text_prompts=perception_config['text_prompts'],
        temperature=1.0  # Start with no calibration
    )
    
    # Create camera params
    camera_config = params['camera']
    camera_params = {
        'intrinsics': create_camera_intrinsics(
            camera_config['resolution'],
            camera_config['fov']
        ),
        'extrinsics': create_camera_extrinsics(
            camera_config['position'],
            camera_config['orientation']
        )
    }
    
    # Fit temperature
    best_T, metrics = fit_temperature(calibration_data, perception, camera_params)
    
    # Save results
    save_calibration_config(best_T, metrics, args.output)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n✓ Calibration complete!")
    print(f"  Temperature: {best_T:.3f}")
    print(f"  ECE: {metrics['ECE_before']:.4f} → {metrics['ECE_after']:.4f}")
    print("\nNext step: Run main.py to test the system")


if __name__ == "__main__":
    import os
    main()
