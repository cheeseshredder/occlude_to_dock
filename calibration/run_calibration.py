"""
Calibration Dataset Collection for ECE Computation.

Collects detection predictions with ground truth labels to compute
Expected Calibration Error (ECE) and fit optimal temperature.
"""

import os
import sys
import numpy as np
import pickle
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def collect_calibration_data_from_scene(
    world,
    robot,
    robot_interface,
    perception,
    objects,
    get_camera_world_pos,
    n_samples: int = 20,
    motion_controller=None,
) -> List[Dict]:
    """
    Collect calibration samples from a single scene by moving robot to different viewpoints.
    
    Returns list of (detections, ground_truth_bboxes, robot_pose) tuples.
    """
    samples = []
    
    # Camera parameters for bbox projection
    camera_params = {
        'width': 640,
        'height': 480,
        'fov': 60.0
    }
    
    # Get ground truth positions
    gt_positions = {}
    object_sizes = {'charger': 0.3, 'plant': 0.4, 'obstacle': 0.35}
    
    for name, obj in objects.items():
        if obj is not None:
            try:
                pos, _ = obj.get_world_pose()
                gt_positions[name] = (float(pos[0]), float(pos[1]))
            except:
                pass
    
    # Generate viewpoints around the scene
    viewpoints = []
    
    # Grid of viewpoints
    for x in np.linspace(-0.5, 2.0, 6):
        for y in np.linspace(-1.0, 1.0, 5):
            # Face toward objects area
            yaw = np.arctan2(0 - y, 2.0 - x)
            viewpoints.append((x, y, yaw))
    
    # Random viewpoints
    for _ in range(n_samples - len(viewpoints)):
        x = np.random.uniform(-0.5, 2.5)
        y = np.random.uniform(-1.5, 1.5)
        yaw = np.random.uniform(-np.pi/2, np.pi/2)
        viewpoints.append((x, y, yaw))
    
    # Shuffle and limit
    np.random.shuffle(viewpoints)
    viewpoints = viewpoints[:n_samples]
    
    for idx, (vx, vy, vyaw) in enumerate(viewpoints):
        try:
            # Move robot to viewpoint
            if motion_controller is not None:
                motion_controller.move_to(vx, vy, vyaw, max_steps=200)
            else:
                # Teleport if no motion controller
                from scipy.spatial.transform import Rotation
                quat = Rotation.from_euler('z', vyaw).as_quat()  # [x,y,z,w]
                quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
                robot.set_world_pose([vx, vy, 0.0], quat_wxyz)
                for _ in range(10):
                    world.step(render=True)
            
            # Get robot pose
            pose = robot_interface.get_pose()
            rx, ry, ryaw = pose
            
            # Update perception camera
            cam_pos = get_camera_world_pos(pose)
            perception.update_camera_pose(cam_pos, ryaw)
            
            # Get RGB and run detection
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            # Compute ground truth bboxes for visible objects
            gt_bboxes = {}
            for name, (wx, wy) in gt_positions.items():
                # Check if visible from current viewpoint
                dx = wx - rx
                dy = wy - ry
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < 0.3 or distance > 5.0:
                    continue
                
                # Check angle
                angle = np.arctan2(dy, dx) - ryaw
                while angle > np.pi: angle -= 2*np.pi
                while angle < -np.pi: angle += 2*np.pi
                
                if abs(angle) > np.radians(35):  # 70° FOV / 2 with margin
                    continue
                
                # Project to image
                from evaluation.calibration_metrics import world_to_image_bbox
                
                obj_size = object_sizes.get(name.split('_')[0], 0.35)
                bbox = world_to_image_bbox((wx, wy), obj_size, pose, camera_params)
                
                if bbox is not None:
                    # Map to class name
                    if 'charger' in name:
                        cls = 'charger'
                    elif 'plant' in name:
                        cls = 'plant'
                    else:
                        cls = 'obstacle'
                    
                    if cls not in gt_bboxes:
                        gt_bboxes[cls] = []
                    gt_bboxes[cls].append(bbox)
            
            samples.append({
                'detections': detections,
                'gt_bboxes': gt_bboxes,
                'robot_pose': pose,
                'viewpoint_idx': idx,
            })
            
            if (idx + 1) % 5 == 0:
                print(f"    Collected {idx + 1}/{n_samples} samples")
                
        except Exception as e:
            print(f"    ⚠ Error at viewpoint {idx}: {e}")
            continue
    
    return samples


def run_calibration_collection(
    num_scenes: int = 5,
    samples_per_scene: int = 30,
    output_path: str = None,
    headless: bool = False,
):
    """
    Run full calibration data collection across multiple scenes.
    """
    if output_path is None:
        output_path = os.path.join(project_root, 'calibration', 'calibration_dataset.pkl')
    
    print("="*60)
    print("CALIBRATION DATA COLLECTION")
    print("="*60)
    print(f"Scenes: {num_scenes}")
    print(f"Samples per scene: {samples_per_scene}")
    print(f"Total samples: {num_scenes * samples_per_scene}")
    print(f"Output: {output_path}")
    print()
    
    # Start Isaac Sim
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": headless, "width": 1280, "height": 720})
    
    all_samples = []
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from perception.vision_model import PerceptionModule, create_isaac_raycast_func
        from omni.physx import get_physx_scene_query_interface
        
        # Camera offset
        CAMERA_OFFSET_X = 0.05
        CAMERA_HEIGHT = 0.20
        
        def get_camera_world_pos(robot_pose):
            rx, ry, ryaw = robot_pose
            cos_yaw = np.cos(ryaw)
            sin_yaw = np.sin(ryaw)
            cam_x = rx + CAMERA_OFFSET_X * cos_yaw
            cam_y = ry + CAMERA_OFFSET_X * sin_yaw
            return np.array([cam_x, cam_y, CAMERA_HEIGHT])
        
        for scene_idx in range(num_scenes):
            seed = 1000 + scene_idx * 17  # Different seeds
            print(f"\n[Scene {scene_idx + 1}/{num_scenes}] seed={seed}")
            
            # Setup scene
            scene = setup_scene('partial', seed=seed)
            world = scene['world']
            robot = scene['robot']
            objects = scene['objects']
            camera_path = scene['camera_path']
            
            # Initialize interfaces
            robot_interface = IsaacSimRobot(
                robot=robot, world=world,
                camera_path=camera_path, objects=objects
            )
            
            # Initialize perception (T=1.0 for raw confidences)
            perception = PerceptionModule(
                model_name="IDEA-Research/grounding-dino-tiny",
                text_prompts={
                    'charger': 'green box . green cube',
                    'plant': 'red box . red cube',
                    'obstacle': 'gray box . gray cube',
                },
                temperature=1.0,  # No calibration - raw confidences
                threshold=0.15,   # Lower threshold to catch more detections
                use_depth_sensor=True,
                depth_noise_enabled=True,
            )
            
            # Setup raycast
            physx_query = get_physx_scene_query_interface()
            raycast_func = create_isaac_raycast_func(physx_query)
            initial_pose = robot_interface.get_pose()
            perception.set_raycast_func(raycast_func, get_camera_world_pos(initial_pose))
            
            # Initialize world
            world.reset()
            for _ in range(30):
                world.step(render=True)
            
            # Collect samples
            samples = collect_calibration_data_from_scene(
                world=world,
                robot=robot,
                robot_interface=robot_interface,
                perception=perception,
                objects=objects,
                get_camera_world_pos=get_camera_world_pos,
                n_samples=samples_per_scene,
            )
            
            # Add scene metadata
            for s in samples:
                s['scene_idx'] = scene_idx
                s['scene_seed'] = seed
            
            all_samples.extend(samples)
            print(f"  ✓ Collected {len(samples)} samples (total: {len(all_samples)})")
            
            # Clear for next scene
            world.clear()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()
    
    # Save dataset
    if all_samples:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(all_samples, f)
        print(f"\n✓ Saved {len(all_samples)} samples to {output_path}")
    
    return all_samples


def evaluate_calibration(dataset_path: str = None) -> Dict:
    """
    Evaluate calibration metrics on collected dataset.
    """
    if dataset_path is None:
        dataset_path = os.path.join(project_root, 'calibration', 'calibration_dataset.pkl')
    
    print("="*60)
    print("CALIBRATION EVALUATION")
    print("="*60)
    
    # Load dataset
    with open(dataset_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"Loaded {len(samples)} samples")
    
    # Import evaluator
    from evaluation.calibration_metrics import CalibrationEvaluator
    
    evaluator = CalibrationEvaluator(iou_threshold=0.5)
    
    # Process all samples
    n_detections = 0
    n_matched = 0
    
    for sample in samples:
        detections = sample['detections']
        gt_bboxes = sample['gt_bboxes']
        
        for det in detections:
            cls = det['class']
            if cls in gt_bboxes and gt_bboxes[cls]:
                evaluator.add_prediction(det, gt_bboxes[cls], cls)
                n_detections += 1
    
    print(f"Processed {n_detections} detections")
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    print(f"\n📊 Calibration Metrics (T=1.0):")
    print(f"   ECE: {metrics['ECE']:.4f}")
    print(f"   MCE: {metrics['MCE']:.4f}")
    print(f"   Brier: {metrics['Brier']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Mean Confidence: {metrics['mean_confidence']:.2%}")
    
    # Find optimal temperature
    best_t, calib_metrics = evaluator.find_optimal_temperature()
    
    print(f"\n🔧 Temperature Calibration:")
    print(f"   Optimal T: {best_t:.3f}")
    print(f"   ECE before: {calib_metrics['ECE_before']:.4f}")
    print(f"   ECE after: {calib_metrics['ECE_after']:.4f}")
    print(f"   Improvement: {calib_metrics['improvement_percent']:.1f}%")
    
    # Per-class metrics
    if metrics.get('per_class'):
        print(f"\n📋 Per-Class Metrics:")
        for cls, cls_metrics in metrics['per_class'].items():
            print(f"   {cls}:")
            print(f"      ECE: {cls_metrics['ECE']:.4f}")
            print(f"      Accuracy: {cls_metrics['accuracy']:.2%}")
            print(f"      Samples: {cls_metrics['n_samples']}")
    
    # Save results
    results = {
        'metrics': metrics,
        'calibration': calib_metrics,
        'optimal_temperature': best_t,
        'timestamp': datetime.now().isoformat(),
    }
    
    results_path = dataset_path.replace('.pkl', '_results.yaml')
    import yaml
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\n✓ Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibration data collection and evaluation')
    parser.add_argument('--collect', action='store_true', help='Collect calibration dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate calibration')
    parser.add_argument('--num_scenes', type=int, default=5)
    parser.add_argument('--samples_per_scene', type=int, default=30)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    
    args = parser.parse_args()
    
    if args.collect:
        run_calibration_collection(
            num_scenes=args.num_scenes,
            samples_per_scene=args.samples_per_scene,
            output_path=args.dataset,
            headless=args.headless,
        )
    
    if args.evaluate:
        evaluate_calibration(args.dataset)
    
    if not args.collect and not args.evaluate:
        print("Usage:")
        print("  Collect data:  python run_calibration.py --collect --num_scenes 5")
        print("  Evaluate:      python run_calibration.py --evaluate")
        print("  Both:          python run_calibration.py --collect --evaluate")
