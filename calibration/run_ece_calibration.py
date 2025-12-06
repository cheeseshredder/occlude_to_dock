"""
ECE Calibration - WITH PLATT SCALING

Temperature scaling is limited - it can only scale confidences symmetrically around 0.5.
Platt scaling uses logistic regression to learn optimal mapping from raw confidence to calibrated probability.
"""

import os
import sys
import numpy as np
import pickle
import yaml
from datetime import datetime
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.calibration_metrics import compute_ECE, compute_MCE, compute_brier_score


def platt_scaling_fit(confidences: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters: P(correct) = sigmoid(a * logit(conf) + b)
    
    Returns (a, b) parameters.
    """
    # Convert confidences to logits
    eps = 1e-7
    conf_clipped = np.clip(confidences, eps, 1 - eps)
    logits = np.log(conf_clipped / (1 - conf_clipped))
    
    def neg_log_likelihood(params):
        a, b = params
        scaled_logits = a * logits + b
        probs = expit(scaled_logits)
        probs = np.clip(probs, eps, 1 - eps)
        # Binary cross entropy
        nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        return nll
    
    # Initialize with a=1, b=0 (identity)
    result = minimize(neg_log_likelihood, [1.0, 0.0], method='Nelder-Mead')
    return result.x[0], result.x[1]


def platt_scaling_apply(confidences: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Platt scaling: P = sigmoid(a * logit(conf) + b)"""
    eps = 1e-7
    conf_clipped = np.clip(confidences, eps, 1 - eps)
    logits = np.log(conf_clipped / (1 - conf_clipped))
    scaled_logits = a * logits + b
    return expit(scaled_logits)


def teleport_robot(robot, world, x, y, yaw, settle_steps=20):
    """Teleport robot to position."""
    quat = Rotation.from_euler('z', yaw).as_quat()
    quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
    robot.set_world_pose([x, y, 0.0], quat_wxyz)
    for _ in range(settle_steps):
        world.step(render=True)


def collect_calibration_data(
    world, robot, robot_interface, perception, objects, get_camera_world_pos,
    n_viewpoints: int = 60,
) -> List[Dict]:
    """Collect calibration data."""
    print(f"\n📸 Collecting calibration data...")
    
    # Ground truth
    gt_positions = {}
    for name, obj in objects.items():
        if obj is not None:
            try:
                pos, _ = obj.get_world_pose()
                gt_positions[name] = (float(pos[0]), float(pos[1]))
            except:
                pass
    
    gt_by_class = {
        'charger': gt_positions.get('charger'),
        'plant': gt_positions.get('plant'),
    }
    
    print(f"   GT charger: {gt_by_class['charger']}")
    print(f"   GT plant: {gt_by_class['plant']}")
    
    # Generate diverse viewpoints
    viewpoints = []
    
    # Dense coverage facing objects
    for dist in [0.8, 1.2, 1.6, 2.0, 2.4]:
        for angle_deg in np.linspace(-50, 50, 9):
            angle = np.radians(angle_deg)
            target = (2.3, 0.0)
            vx = target[0] - dist * np.cos(angle)
            vy = target[1] - dist * np.sin(angle)
            vyaw = angle
            if 0.0 <= vx <= 2.2 and -1.5 <= vy <= 1.5:
                viewpoints.append((vx, vy, vyaw))
    
    viewpoints = list(set(viewpoints))
    np.random.shuffle(viewpoints)
    viewpoints = viewpoints[:n_viewpoints]
    print(f"   Using {len(viewpoints)} viewpoints")
    
    samples = []
    
    for idx, (vx, vy, vyaw) in enumerate(viewpoints):
        try:
            teleport_robot(robot, world, vx, vy, vyaw)
            
            pose = robot_interface.get_pose()
            cam_pos = get_camera_world_pos(pose)
            perception.update_camera_pose(cam_pos, pose[2])
            
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            for det in detections:
                det_class = det['class']
                if det_class == 'obstacle':
                    continue
                
                det_conf = det.get('confidence', 0.5)
                det_world = det.get('world_position')
                
                if det_world is None:
                    continue
                
                gt_pos = gt_by_class.get(det_class)
                if gt_pos is None:
                    continue
                
                dist_error = np.sqrt(
                    (det_world[0] - gt_pos[0])**2 + 
                    (det_world[1] - gt_pos[1])**2
                )
                
                is_correct = dist_error < 1.0
                
                samples.append({
                    'confidence': det_conf,
                    'is_correct': is_correct,
                    'class': det_class,
                    'distance_error': dist_error,
                })
            
            if (idx + 1) % 15 == 0:
                n_correct = sum(1 for s in samples if s['is_correct'])
                print(f"   Viewpoint {idx + 1}: {len(samples)} samples, {n_correct} correct")
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    n_correct = sum(1 for s in samples if s['is_correct'])
    print(f"   ✅ Total: {len(samples)} samples, {n_correct} correct ({100*n_correct/max(1,len(samples)):.1f}%)")
    
    return samples


def evaluate_calibration(samples: List[Dict]) -> Dict:
    """Evaluate with both temperature and Platt scaling."""
    print(f"\n📊 Evaluating calibration...")
    
    if len(samples) < 15:
        print("   ⚠️ Too few samples!")
        return None
    
    confidences = np.array([s['confidence'] for s in samples])
    labels = np.array([1 if s['is_correct'] else 0 for s in samples])
    
    accuracy = np.mean(labels)
    mean_conf = np.mean(confidences)
    
    print(f"   Samples: {len(samples)}")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Mean confidence: {mean_conf:.1%}")
    print(f"   Gap: {abs(accuracy - mean_conf):.1%} ({'underconfident' if mean_conf < accuracy else 'overconfident'})")
    
    # === BEFORE CALIBRATION ===
    ece_before = compute_ECE(confidences, labels)
    mce_before = compute_MCE(confidences, labels)
    brier_before = compute_brier_score(confidences, labels)
    
    print(f"\n📈 BEFORE calibration:")
    print(f"   ECE: {ece_before:.4f}")
    print(f"   MCE: {mce_before:.4f}")
    print(f"   Brier: {brier_before:.4f}")
    
    # === PLATT SCALING ===
    print(f"\n🔧 Fitting Platt scaling...")
    a, b = platt_scaling_fit(confidences, labels)
    print(f"   Parameters: a={a:.3f}, b={b:.3f}")
    
    calibrated_platt = platt_scaling_apply(confidences, a, b)
    ece_platt = compute_ECE(calibrated_platt, labels)
    mce_platt = compute_MCE(calibrated_platt, labels)
    brier_platt = compute_brier_score(calibrated_platt, labels)
    
    print(f"\n📈 AFTER Platt scaling:")
    print(f"   ECE: {ece_platt:.4f} (was {ece_before:.4f})")
    print(f"   MCE: {mce_platt:.4f}")
    print(f"   Brier: {brier_platt:.4f}")
    print(f"   Mean calibrated conf: {np.mean(calibrated_platt):.1%}")
    print(f"   Improvement: {100*(ece_before - ece_platt)/ece_before:.1f}%")
    
    # === PER-CLASS ===
    print(f"\n📋 Per-class (after Platt):")
    for cls in ['charger', 'plant']:
        cls_idx = [i for i, s in enumerate(samples) if s['class'] == cls]
        if len(cls_idx) >= 5:
            cls_conf = calibrated_platt[cls_idx]
            cls_labels = labels[cls_idx]
            cls_ece = compute_ECE(cls_conf, cls_labels)
            print(f"   {cls}: ECE={cls_ece:.4f}, Acc={np.mean(cls_labels):.1%}, N={len(cls_idx)}")
    
    return {
        'metrics': {
            'ECE_before': float(ece_before),
            'ECE_after': float(ece_platt),
            'MCE': float(mce_platt),
            'Brier': float(brier_platt),
            'accuracy': float(accuracy),
            'mean_confidence_raw': float(mean_conf),
            'mean_confidence_calibrated': float(np.mean(calibrated_platt)),
            'n_samples': len(samples),
        },
        'platt_params': {
            'a': float(a),
            'b': float(b),
        },
        'improvement_percent': float(100*(ece_before - ece_platt)/ece_before),
    }


def run_calibration(n_viewpoints: int = 60, headless: bool = False):
    """Run calibration with Platt scaling."""
    print("="*60)
    print("ECE CALIBRATION - PLATT SCALING")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": headless, "width": 1280, "height": 720})
    
    results = None
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from perception.vision_model import PerceptionModule, create_isaac_raycast_func
        from omni.physx import get_physx_scene_query_interface
        
        CAMERA_OFFSET_X = 0.05
        CAMERA_HEIGHT = 0.20
        
        def get_camera_world_pos(robot_pose):
            rx, ry, ryaw = robot_pose
            return np.array([
                rx + CAMERA_OFFSET_X * np.cos(ryaw),
                ry + CAMERA_OFFSET_X * np.sin(ryaw),
                CAMERA_HEIGHT
            ])
        
        print("\n[1/4] Setting up scene...")
        scene = setup_scene('partial', seed=42)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        camera_path = scene['camera_path']
        
        print("\n[2/4] Initializing robot interface...")
        robot_interface = IsaacSimRobot(
            robot=robot, world=world,
            camera_path=camera_path, objects=objects
        )
        
        print("\n[3/4] Initializing perception...")
        perception = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompts={
                'charger': 'green box . green cube . green object',
                'plant': 'red box . red cube . red object',
                'obstacle': 'gray box . gray cube',
            },
            temperature=1.0,  # Raw confidences
            threshold=0.20,
            use_depth_sensor=True,
            depth_noise_enabled=True,
        )
        
        physx_query = get_physx_scene_query_interface()
        raycast_func = create_isaac_raycast_func(physx_query)
        initial_pose = robot_interface.get_pose()
        perception.set_raycast_func(raycast_func, get_camera_world_pos(initial_pose))
        
        world.reset()
        for _ in range(30):
            world.step(render=True)
        
        print("\n[4/4] Collecting data...")
        samples = collect_calibration_data(
            world, robot, robot_interface, perception, objects,
            get_camera_world_pos, n_viewpoints
        )
        
        results = evaluate_calibration(samples)
        
        if results:
            # Save
            output_dir = os.path.join(project_root, 'calibration')
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'calibration_samples.pkl'), 'wb') as f:
                pickle.dump(samples, f)
            
            results['timestamp'] = datetime.now().isoformat()
            with open(os.path.join(output_dir, 'calibration_results.yaml'), 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            
            # Config for perception module
            config_dir = os.path.join(project_root, 'config')
            os.makedirs(config_dir, exist_ok=True)
            config = {
                'calibration_method': 'platt',
                'platt_a': results['platt_params']['a'],
                'platt_b': results['platt_params']['b'],
                'ECE_before': results['metrics']['ECE_before'],
                'ECE_after': results['metrics']['ECE_after'],
                'accuracy': results['metrics']['accuracy'],
                'n_samples': results['metrics']['n_samples'],
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(os.path.join(config_dir, 'calibration.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print("\n" + "="*60)
            print("CALIBRATION COMPLETE")
            print("="*60)
            print(f"Platt params: a={results['platt_params']['a']:.3f}, b={results['platt_params']['b']:.3f}")
            print(f"ECE: {results['metrics']['ECE_before']:.4f} → {results['metrics']['ECE_after']:.4f}")
            print(f"Accuracy: {results['metrics']['accuracy']:.1%}")
            print(f"Improvement: {results['improvement_percent']:.1f}%")
            
            ece_after = results['metrics']['ECE_after']
            if ece_after < 0.08:
                print("\n✅ ECE < 0.08 - EXCELLENT!")
            elif ece_after < 0.10:
                print("\n✅ ECE < 0.10 - GOOD")
            elif ece_after < 0.15:
                print("\n⚠️ ECE < 0.15 - ACCEPTABLE")
            else:
                print("\n❌ ECE >= 0.15 - May need more data")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpoints', type=int, default=60)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    
    n = 25 if args.quick else args.viewpoints
    run_calibration(n_viewpoints=n, headless=args.headless)
