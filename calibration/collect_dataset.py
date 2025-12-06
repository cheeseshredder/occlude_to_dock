"""
Calibration Dataset Collection with REAL camera capture
"""

import os
import sys
import numpy as np
import pickle
import argparse
from typing import List, Dict
from datetime import datetime

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def collect_calibration_dataset(simulation_app, num_scenes: int = 10, 
                                samples_per_scene: int = 20) -> List[Dict]:
    """Collect calibration dataset with REAL images from Isaac Sim."""
    
    from scene.setup_environment import setup_scene
    from camera_utils import get_camera_data, get_ground_truth
    
    print(f"Collecting calibration dataset...")
    print(f"  Scenes: {num_scenes}")
    print(f"  Samples per scene: {samples_per_scene}")
    print(f"  Total samples: {num_scenes * samples_per_scene}\n")
    
    calibration_data = []
    
    for scene_id in range(num_scenes):
        seed = 1000 + scene_id
        print(f"Scene {scene_id+1}/{num_scenes} (seed={seed})")
        
        # CRITICAL: Clear previous scene completely
        if scene_id > 0:
            print("  → Clearing previous scene...")
            from omni.isaac.core import World
            World.clear_instance()  # Removes all objects!
            
            # Reload the module for fresh imports
            import importlib
            import scene.setup_environment as setup_env
            importlib.reload(setup_env)
        
        try:
            # Setup scene
            scene_dict = setup_scene('partial', seed=seed)
            world = scene_dict['world']
            camera = scene_dict['camera']
            objects = scene_dict['objects']
            robot = scene_dict['robot']
            
            # Collect samples from this scene
            for sample_id in range(samples_per_scene):
                
                # Move robot slightly for variety
                if sample_id > 0 and robot:
                    try:
                        current_pos, current_rot = robot.get_world_pose()
                        noise_x = np.random.uniform(-0.2, 0.2)
                        noise_y = np.random.uniform(-0.2, 0.2)
                        new_pos = [current_pos[0] + noise_x, current_pos[1] + noise_y, current_pos[2]]
                        robot.set_world_pose(new_pos, current_rot)
                        
                        # Step to update
                        for _ in range(5):
                            world.step(render=True)
                    except:
                        pass
                
                # CRITICAL: Capture REAL images using replicator
                rgb, depth = get_camera_data(camera, world)
                
                # Verify we got real data
                if rgb.sum() == 0:
                    print(f"  ⚠ Sample {sample_id}: Got black image!")
                
                # Get ground truth
                ground_truth = get_ground_truth(world, objects, camera)
                
                # Get robot pose
                if robot:
                    try:
                        robot_pose_3d, robot_rot = robot.get_world_pose()
                        import math
                        siny_cosp = 2 * (robot_rot[0] * robot_rot[3] + robot_rot[1] * robot_rot[2])
                        cosy_cosp = 1 - 2 * (robot_rot[2]**2 + robot_rot[3]**2)
                        yaw = math.atan2(siny_cosp, cosy_cosp)
                        robot_pose = np.array([robot_pose_3d[0], robot_pose_3d[1], yaw])
                    except:
                        robot_pose = np.array([0.0, 0.0, 0.0])
                else:
                    robot_pose = np.array([0.0, 0.0, 0.0])

                calibration_data.append({
                    'rgb': rgb,
                    'depth': depth,
                    'ground_truth': ground_truth,
                    'robot_pose': robot_pose,
                    'scene_id': scene_id,
                    'sample_id': sample_id
                })
                
                if (sample_id + 1) % 5 == 0:
                    print(f"  Collected {sample_id + 1}/{samples_per_scene} samples")
                    
        except Exception as e:
            print(f"❌ Error in scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Collection complete: {len(calibration_data)} samples")
    
    # Verify data quality
    non_zero_count = sum(1 for sample in calibration_data if sample['rgb'].sum() > 0)
    print(f"  → Real images (non-zero): {non_zero_count}/{len(calibration_data)}")
    
    if non_zero_count == 0:
        print("\n⚠️  WARNING: All images are black! Camera capture may have failed.")
    
    return calibration_data


def save_calibration_dataset(data: List[Dict], filepath: str):
    """Save calibration dataset to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Dataset saved to: {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Collect calibration dataset')
    parser.add_argument('--num_scenes', type=int, default=10)
    parser.add_argument('--samples_per_scene', type=int, default=20)
    parser.add_argument('--output', type=str, default='calibration/calibration_dataset.pkl')
    
    args = parser.parse_args()
    
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        print("❌ ERROR: Isaac Sim not available")
        sys.exit(1)
    
    print("="*60)
    print("Calibration Dataset Collection - WITH REAL IMAGES")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Starting Isaac Sim...")
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    print("✓ Isaac Sim started\n")
    
    try:
        dataset = collect_calibration_dataset(
            simulation_app,
            num_scenes=args.num_scenes,
            samples_per_scene=args.samples_per_scene
        )
        
        save_calibration_dataset(dataset, args.output)
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print("\n✅ Dataset collection complete!")
        print(f"   Samples: {len(dataset)}")
        print(f"   Output: {args.output}")
        print("\nNext step: Run fit_temperature.py to calibrate")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR during data collection")
        print("="*60)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nClosing Isaac Sim...")
        simulation_app.close()


if __name__ == "__main__":
    main()