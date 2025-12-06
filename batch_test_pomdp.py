"""
Batch Testing Script for POMDP Navigation

Runs multiple trials in headless mode and collects:
- Success/failure status
- Time to dock
- Final distance
- Navigation steps
- Belief maps for each step

Usage:
    isaac-python.bat batch_test_pomdp.py --trials 5 --headless
    isaac-python.bat batch_test_pomdp.py --trials 3  # With GUI for debugging
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Parse arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description='Batch test POMDP navigation')
parser.add_argument('--trials', type=int, default=5, help='Number of test trials')
parser.add_argument('--headless', action='store_true', help='Run without GUI')
parser.add_argument('--seed-start', type=int, default=42, help='Starting random seed')
parser.add_argument('--output-dir', type=str, default='batch_results', help='Output directory')
parser.add_argument('--extra-warmup', type=int, default=100, help='Extra warmup steps for headless')
parser.add_argument('--tier', type=str, default='heavy', choices=['partial', 'heavy'],
                    help='Occlusion tier: "heavy" (charger hidden) or "partial" (charger edges visible)')
parser.add_argument('--instruction', type=str, default='dock at the charger behind the plant',
                    help='Natural language instruction for the docking task')
args = parser.parse_args()

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create output directory
output_dir = os.path.join(project_root, args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Timestamp for this batch run
batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
batch_dir = os.path.join(output_dir, f"batch_{args.tier}_{batch_timestamp}")
os.makedirs(batch_dir, exist_ok=True)

print("="*60)
print("POMDP NAVIGATION BATCH TESTING")
print("="*60)
print(f"Trials: {args.trials}")
print(f"Headless: {args.headless}")
print(f"Seed start: {args.seed_start}")
print(f"Output: {batch_dir}")
print(f"Extra warmup steps: {args.extra_warmup}")
print(f"Occlusion tier: {args.tier}")
print(f"Instruction: {args.instruction}")
print("="*60)

# Initialize Isaac Sim
from omni.isaac.kit import SimulationApp

simulation_config = {
    "headless": args.headless,
    "width": 1280,
    "height": 720
}
simulation_app = SimulationApp(simulation_config)

# Now import everything else
import numpy as np
from PIL import Image, ImageDraw

# Import our modules
from scene.setup_environment import setup_scene
from scene.robot_interface import IsaacSimRobot
from perception.apriltag_detector import HybridPerceptionModule, AprilTagChargerDetector
from perception.vision_model import PerceptionModule, create_isaac_raycast_func
from mapping.semantic_grid import SemanticGrid
from planner.pomdp_planner import POMDPPlanner
from omni.isaac.core.utils.types import ArticulationAction
from omni.physx import get_physx_scene_query_interface

# Import language parser
from language.parser import LanguageParser, ParsedInstruction

# Parse the instruction
language_parser = LanguageParser()
parsed_instruction = language_parser.parse(args.instruction)
detection_prompts = language_parser.get_detection_prompts(parsed_instruction)
planner_config = language_parser.get_planner_config(parsed_instruction)

print("\n" + "="*60)
print("PARSED INSTRUCTION")
print("="*60)
print(parsed_instruction)
print(f"\nDetection Prompts: {detection_prompts}")
print(f"Planner Config: {planner_config}")
print("="*60 + "\n")


def is_path_blocked_by_plant(robot_pos, charger_pos, plant_pos, clearance=0.4):
    """
    Check if plant blocks the straight-line path from robot to charger.
    
    Uses geometric line-segment intersection:
    1. Compute path vector from robot to charger
    2. Project plant position onto this path
    3. Check if projection falls between robot and charger
    4. Check perpendicular distance from plant to path
    
    Args:
        robot_pos: (rx, ry) robot position
        charger_pos: (cx, cy) charger position  
        plant_pos: (px, py) plant position
        clearance: minimum distance from plant to path to be considered "clear"
        
    Returns:
        True if plant blocks path, False otherwise
    """
    rx, ry = robot_pos
    cx, cy = charger_pos
    px, py = plant_pos
    
    # Vector from robot to charger
    dx = cx - rx
    dy = cy - ry
    path_length = np.sqrt(dx**2 + dy**2)
    
    if path_length < 0.01:  # Robot already at charger
        return False
    
    # Normalize direction
    dx_norm = dx / path_length
    dy_norm = dy / path_length
    
    # Vector from robot to plant
    px_rel = px - rx
    py_rel = py - ry
    
    # Project plant onto path line (dot product)
    projection = px_rel * dx_norm + py_rel * dy_norm
    
    # Is plant between robot and charger?
    if projection < 0:  # Plant is behind robot
        return False
    if projection > path_length:  # Plant is past charger
        return False
    
    # Perpendicular distance from plant to path line
    # Using cross product: |a × b| = |a||b|sin(θ)
    perp_dist = abs(px_rel * dy_norm - py_rel * dx_norm)
    
    # Plant blocks if it's close to the path
    return perp_dist < clearance


class SmoothMotionController:
    """Simplified motion controller for batch testing."""
    
    WHEEL_RADIUS = 0.033
    WHEEL_BASE = 0.16
    MAX_LINEAR_VEL = 0.08
    MAX_ANGULAR_VEL = 0.6
    KP_LINEAR = 0.5
    KP_ANGULAR = 1.2
    POSITION_TOLERANCE = 0.15
    ANGLE_TOLERANCE = 0.1
    
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world
        self.ArticulationAction = ArticulationAction
        
    def get_pose(self):
        pos, quat = self.robot.get_world_pose()
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return float(pos[0]), float(pos[1]), float(yaw)
    
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def set_wheel_velocities(self, v_linear, v_angular):
        v_linear = np.clip(v_linear, -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        v_angular = np.clip(v_angular, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        v_left = (v_linear - v_angular * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        v_right = (v_linear + v_angular * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        self.robot.apply_wheel_actions(
            self.ArticulationAction(joint_velocities=np.array([v_left, v_right]))
        )
    
    def stop(self):
        self.set_wheel_velocities(0.0, 0.0)
        
    def move_to(self, target_x, target_y, target_yaw=None, max_steps=500):
        position_history = []
        HISTORY_WINDOW = 100
        MIN_MOVEMENT = 0.05
        
        for step in range(max_steps):
            x, y, yaw = self.get_pose()
            position_history.append((x, y))
            if len(position_history) > HISTORY_WINDOW:
                position_history.pop(0)
            
            dx = target_x - x
            dy = target_y - y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Stuck detection
            if step > 0 and step % 50 == 0 and len(position_history) >= HISTORY_WINDOW:
                oldest = position_history[0]
                movement = np.sqrt((x - oldest[0])**2 + (y - oldest[1])**2)
                if movement < MIN_MOVEMENT:
                    self.stop()
                    for _ in range(30):
                        self.world.step(render=True)
                    return False, step
            
            if distance < self.POSITION_TOLERANCE:
                self.stop()
                if target_yaw is not None:
                    self._rotate_to(target_yaw)
                for _ in range(30):
                    self.world.step(render=True)
                return True, step
            
            target_heading = np.arctan2(dy, dx)
            heading_error = self.normalize_angle(target_heading - yaw)
            
            if abs(heading_error) > 0.3:
                v_linear = 0.0
                v_angular = self.KP_ANGULAR * heading_error
            else:
                v_linear = self.KP_LINEAR * min(distance, 0.5)
                v_angular = self.KP_ANGULAR * heading_error
            
            self.set_wheel_velocities(v_linear, v_angular)
            self.world.step(render=True)
        
        self.stop()
        for _ in range(30):
            self.world.step(render=True)
        return False, max_steps
    
    def _rotate_to(self, target_yaw, max_steps=200):
        for step in range(max_steps):
            x, y, yaw = self.get_pose()
            yaw_error = self.normalize_angle(target_yaw - yaw)
            
            if abs(yaw_error) < self.ANGLE_TOLERANCE:
                self.stop()
                for _ in range(30):
                    self.world.step(render=True)
                return True
            
            v_angular = self.KP_ANGULAR * yaw_error
            if abs(yaw_error) > 0.2 and abs(v_angular) < 0.5:
                v_angular = 0.5 * np.sign(yaw_error)
            
            self.set_wheel_velocities(0.0, v_angular)
            self.world.step(render=True)
        
        self.stop()
        for _ in range(30):
            self.world.step(render=True)
        return False
    
    def backup(self, distance=0.3, max_steps=300):
        start_x, start_y, _ = self.get_pose()
        
        for step in range(max_steps):
            x, y, _ = self.get_pose()
            dist_moved = np.sqrt((x - start_x)**2 + (y - start_y)**2)
            
            if dist_moved >= distance:
                self.stop()
                for _ in range(30):
                    self.world.step(render=True)
                return True, step
            
            self.set_wheel_velocities(-self.MAX_LINEAR_VEL, 0.0)
            self.world.step(render=True)
        
        self.stop()
        for _ in range(30):
            self.world.step(render=True)
        dist_moved = np.sqrt((self.get_pose()[0] - start_x)**2 + (self.get_pose()[1] - start_y)**2)
        return dist_moved > 0.05, max_steps


def run_single_trial(trial_num, seed, trial_dir, extra_warmup=0, world_to_clear=None):
    """Run a single navigation trial and return results."""
    
    print(f"\n{'='*60}")
    print(f"TRIAL {trial_num} (seed={seed}, tier={args.tier})")
    print(f"{'='*60}")
    
    trial_results = {
        'trial': trial_num,
        'seed': seed,
        'instruction': args.instruction,
        'parsed_instruction': parsed_instruction.to_dict(),
        'success': False,
        'time_to_dock': None,
        'final_distance': None,
        'nav_steps': 0,
        'plant_blocking_triggered': False,
        'dock_failures': 0,
        'error': None,
        'belief_maps': []
    }
    
    try:
        # CRITICAL: Clear previous world/stage before creating new scene
        if world_to_clear is not None:
            try:
                print("  Clearing previous world...")
                world_to_clear.stop()
                world_to_clear.clear()
                # Note: Async viewport camera errors may appear - they are harmless
            except Exception as e:
                print(f"  (Cleanup warning - non-fatal: {e})")
        
        # Clear the entire USD stage for a fresh start
        import omni.usd
        from pxr import Usd, Sdf
        
        context = omni.usd.get_context()
        
        # Create a completely new stage
        print("  Creating fresh USD stage...")
        context.new_stage()
        
        # Wait for stage to be ready
        import time
        time.sleep(0.5)
        
        # Setup scene - this creates a fresh World instance
        scene = setup_scene(args.tier, seed=seed, use_apriltag=True)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        camera_path = scene['camera_path']
        
        # Store world ref for cleanup (internal, not saved to JSON)
        trial_results['_world_ref'] = world
        
        robot_interface = IsaacSimRobot(
            robot=robot, world=world,
            camera_path=camera_path, objects=objects
        )
        
        motion_controller = SmoothMotionController(robot, world)
        
        # Perception setup
        actual_hfov_deg = 2 * np.degrees(np.arctan(20.955 / (2 * 24.0)))
        
        # Use detection prompts from parsed instruction
        grounding_dino = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompts=detection_prompts,  # From language parser!
            platt_a=1.509, platt_b=1.598,
            threshold=0.2,
            use_depth_sensor=True,
            depth_noise_enabled=True,
            hfov_deg=actual_hfov_deg,
            img_width=640, img_height=480,
        )
        print(f"  Using parsed detection prompts: {detection_prompts}")
        
        apriltag_detector = AprilTagChargerDetector(
            charger_tag_id=0,
            # tag_size defaults to 0.06 (6cm) for close-range charger detection
            # Wall tags are 15cm but we use known positions anyway
            hfov_deg=None,
            focal_length_mm=24.0,
            horizontal_aperture_mm=20.955,
            img_width=640, img_height=480,
        )
        
        perception = HybridPerceptionModule(
            grounding_dino_detector=grounding_dino,
            apriltag_detector=apriltag_detector,
        )
        
        grid = SemanticGrid(
            size_meters=(6.0, 6.0),
            resolution=0.5,
            origin=(-3.0, -3.0),
            close_range_threshold=1.2,
            close_range_weight_factor=0.2,
            lock_threshold=1.0,
            min_detections_to_lock=2,
        )
        
        # Use confidence threshold from parsed instruction
        planner = POMDPPlanner(
            grid=grid,
            num_candidates=12,
            confidence_threshold=planner_config.get('confidence_threshold', 0.4),
            obstacle_avoidance_radius=0.25,  # Keep tight for narrow passages
            room_bounds=(-2.7, -2.7, 2.7, 2.7),
        )
        print(f"  Planner config: confidence_threshold={planner_config.get('confidence_threshold', 0.4)}, "
              f"expect_occlusion={planner_config.get('expect_occlusion', False)}")
        
        # Initialize simulation with EXTRA warmup for headless
        world.reset()
        warmup_steps = 150 + extra_warmup
        print(f"  Warming up ({warmup_steps} steps)...")
        for _ in range(warmup_steps):
            world.step(render=True)
        
        robot_interface.refresh_viewport()
        
        # Raycast setup
        physx_query = get_physx_scene_query_interface()
        raycast_func = create_isaac_raycast_func(physx_query)
        
        CAMERA_OFFSET_X = 0.05
        CAMERA_HEIGHT = 0.20
        
        def get_camera_world_pos(robot_pose):
            rx, ry, ryaw = robot_pose
            cos_yaw, sin_yaw = np.cos(ryaw), np.sin(ryaw)
            cam_x = rx + CAMERA_OFFSET_X * cos_yaw
            cam_y = ry + CAMERA_OFFSET_X * sin_yaw
            return np.array([cam_x, cam_y, CAMERA_HEIGHT])
        
        initial_pose = robot_interface.get_pose()
        initial_cam_pos = get_camera_world_pos(initial_pose)
        perception.set_raycast_func(raycast_func, initial_cam_pos)
        
        def save_belief_map(step, mode, robot_pose, detections, rgb_frame=None):
            """Save belief map visualization."""
            vis = grid.visualize(robot_pose)
            grid_img = Image.fromarray(vis)
            grid_img = grid_img.resize((400, 400), Image.NEAREST)
            
            if rgb_frame is not None:
                cam_img = Image.fromarray(rgb_frame)
                cam_img = cam_img.resize((400, 300), Image.LANCZOS)
            else:
                cam_img = Image.new('RGB', (400, 300), (50, 50, 50))
            
            # Draw detections on camera image
            cam_draw = ImageDraw.Draw(cam_img)
            colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
            for det in detections:
                x1, y1, x2, y2 = det['bbox_2d']
                x1, x2 = int(x1 * 400 / 640), int(x2 * 400 / 640)
                y1, y2 = int(y1 * 300 / 480), int(y2 * 300 / 480)
                color = colors.get(det['class'], (255, 255, 0))
                cam_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Combine
            combined = Image.new('RGB', (820, 420), (255, 255, 255))
            combined.paste(grid_img, (10, 10))
            combined.paste(cam_img, (420, 10))
            
            draw = ImageDraw.Draw(combined)
            charger_pos, charger_conf = grid.get_charger_estimate()
            plant_pos, _ = grid.get_class_estimate('plant')
            
            draw.text((420, 320), f"Step {step} | Mode: {mode}", fill=(0, 0, 0))
            draw.text((420, 340), f"Robot: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f})", fill=(0, 0, 0))
            draw.text((420, 360), f"Charger conf: {charger_conf:.2f}", fill=(0, 0, 0))
            if charger_pos:
                draw.text((420, 380), f"Charger: ({charger_pos[0]:.2f}, {charger_pos[1]:.2f})", fill=(0, 150, 0))
            if plant_pos:
                draw.text((420, 400), f"Plant: ({plant_pos[0]:.2f}, {plant_pos[1]:.2f})", fill=(200, 0, 0))
            
            save_path = os.path.join(trial_dir, f"step_{step:02d}_{mode}.png")
            combined.save(save_path)
            trial_results['belief_maps'].append(save_path)
            return save_path
        
        # Start timing
        start_time = time.time()
        
        # Initial observation with extra stabilization for headless
        print("  Initial observation...")
        for _ in range(100 + extra_warmup // 2):
            world.step(render=True)
        
        pose = robot_interface.get_pose()
        cam_pos = get_camera_world_pos(pose)
        perception.update_camera_pose(cam_pos, pose[2])
        
        rgb = robot_interface.get_rgb()
        
        # Validate image
        img_mean = np.mean(rgb)
        img_std = np.std(rgb)
        retry_count = 0
        while (img_mean < 100 or img_std < 20) and retry_count < 3:
            print(f"  Image invalid (mean={img_mean:.1f}, std={img_std:.1f}), retrying...")
            for _ in range(100):
                world.step(render=True)
            rgb = robot_interface.get_rgb()
            img_mean = np.mean(rgb)
            img_std = np.std(rgb)
            retry_count += 1
        
        if img_mean < 100 or img_std < 20:
            trial_results['error'] = f"Failed to get valid image after {retry_count} retries"
            return trial_results
        
        detections = perception.detect(rgb, pose)
        grid.update_from_detections(detections, pose)
        save_belief_map(0, "INITIAL", pose, detections, rgb)
        
        # Navigation loop
        max_steps = 25
        step = 0
        success = False
        dock_failure_count = 0
        
        while step < max_steps:
            step += 1
            
            # Stabilization
            motion_controller.stop()
            for _ in range(50):
                world.step(render=True)
            
            pose = robot_interface.get_pose()
            cam_pos = get_camera_world_pos(pose)
            perception.update_camera_pose(cam_pos, pose[2])
            
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            grid.update_from_detections(detections, pose)
            
            # Check for visual dock trigger
            charger_detection = None
            for det in detections:
                if det['class'] == 'charger' and det['estimated_distance'] < 3.0:
                    charger_detection = det
                    break
            
            if charger_detection:
                visual_dist = charger_detection['estimated_distance']
                viewing_angle = charger_detection.get('viewing_angle_deg', 0)
                dock_threshold = 1.0 if viewing_angle > 60 else 2.0
                
                if visual_dist < dock_threshold:
                    rx, ry, ryaw = pose
                    
                    # Get charger position for path check
                    charger_world = charger_detection.get('world_position')
                    if charger_world:
                        cx, cy = charger_world
                    else:
                        cx, cy = 2.8, 0.0  # Default charger position
                    
                    # Check plant blocking using GEOMETRIC path check
                    plant_blocking = False
                    if grid.locked_estimates['plant'] is not None:
                        px, py = grid.locked_estimates['plant']['position']
                        # Use geometric check: is plant on the line between robot and charger?
                        plant_blocking = is_path_blocked_by_plant(
                            robot_pos=(rx, ry),
                            charger_pos=(cx, cy),
                            plant_pos=(px, py),
                            clearance=0.4  # Plant + safety margin
                        )
                        if plant_blocking:
                            trial_results['plant_blocking_triggered'] = True
                            print(f"  🌱 Plant at ({px:.2f}, {py:.2f}) blocking path to charger!")
                        else:
                            print(f"  ✅ Path clear (plant at ({px:.2f}, {py:.2f}) not blocking)")
                    
                    if not plant_blocking:
                        # Attempt dock (cx, cy already set above)
                        approach_dist = max(0.1, visual_dist - 0.1)
                        target_x = rx + approach_dist * np.cos(ryaw)
                        target_y = ry + approach_dist * np.sin(ryaw)
                        
                        reached, _ = motion_controller.move_to(target_x, target_y, max_steps=400)
                        
                        if not reached:
                            dock_failure_count += 1
                            trial_results['dock_failures'] = dock_failure_count
                            
                            if dock_failure_count >= 3:
                                trial_results['error'] = "Too many dock failures"
                                break
                            
                            motion_controller.backup(0.3)
                            continue
                        
                        # Check final distance
                        final_pose = motion_controller.get_pose()
                        final_dist = np.sqrt((final_pose[0] - 2.8)**2 + (final_pose[1] - 0.0)**2)
                        
                        if final_dist < 0.30:
                            success = True
                            trial_results['success'] = True
                            trial_results['error'] = None  # Clear any previous error
                            trial_results['final_distance'] = final_dist
                            trial_results['time_to_dock'] = time.time() - start_time
                            trial_results['nav_steps'] = step
                            
                            save_belief_map(step, "DOCK_SUCCESS", final_pose, detections, rgb)
                            print(f"  ✅ DOCKED! Distance: {final_dist:.2f}m")
                            break
            
            # Get planner action - returns (target, mode) tuple
            target, mode = planner.get_navigation_target(pose)
            
            save_belief_map(step, mode, pose, detections, rgb)
            
            if target:
                motion_controller.move_to(target[0], target[1], max_steps=1000)
        
        if not success:
            final_pose = motion_controller.get_pose()
            final_dist = np.sqrt((final_pose[0] - 2.8)**2 + (final_pose[1] - 0.0)**2)
            trial_results['final_distance'] = final_dist
            trial_results['time_to_dock'] = time.time() - start_time
            trial_results['nav_steps'] = step
            
            # Check if close enough to count as success
            if final_dist < 0.30:
                trial_results['success'] = True
                trial_results['error'] = None  # Clear any previous error
                print(f"  ✅ Close enough! Distance: {final_dist:.2f}m")
            else:
                print(f"  ❌ Failed. Distance: {final_dist:.2f}m")
        
    except Exception as e:
        trial_results['error'] = str(e)
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return trial_results


def main():
    all_results = []
    last_world = None
    
    for trial in range(args.trials):
        seed = args.seed_start + trial
        trial_dir = os.path.join(batch_dir, f"trial_{trial:02d}_seed{seed}")
        os.makedirs(trial_dir, exist_ok=True)
        
        result = run_single_trial(
            trial_num=trial,
            seed=seed,
            trial_dir=trial_dir,
            extra_warmup=args.extra_warmup if args.headless else 0,
            world_to_clear=last_world
        )
        all_results.append(result)
        
        # Store world reference for next trial cleanup
        # Note: world may be None if trial failed during setup
        last_world = result.get('_world_ref', None)
        
        # Save intermediate results (remove non-serializable _world_ref)
        serializable_results = []
        for r in all_results:
            r_copy = {k: v for k, v in r.items() if not k.startswith('_')}
            serializable_results.append(r_copy)
        with open(os.path.join(batch_dir, "results.json"), 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*60)
    print("BATCH TEST SUMMARY")
    print("="*60)
    
    successes = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    print(f"\nSuccess Rate: {successes}/{total} ({100*successes/total:.1f}%)")
    
    successful_times = [r['time_to_dock'] for r in all_results if r['success'] and r['time_to_dock']]
    if successful_times:
        print(f"Avg Time to Dock: {np.mean(successful_times):.1f}s (±{np.std(successful_times):.1f}s)")
    
    successful_dists = [r['final_distance'] for r in all_results if r['success'] and r['final_distance']]
    if successful_dists:
        print(f"Avg Final Distance: {np.mean(successful_dists):.3f}m (±{np.std(successful_dists):.3f}m)")
    
    plant_blocks = sum(1 for r in all_results if r['plant_blocking_triggered'])
    print(f"Plant Blocking Triggered: {plant_blocks}/{total} trials")
    
    errors = [r for r in all_results if r['error']]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  Trial {r['trial']}: {r['error']}")
    
    print(f"\nResults saved to: {batch_dir}")
    
    # Save final summary
    summary = {
        'batch_timestamp': batch_timestamp,
        'instruction': args.instruction,
        'parsed_instruction': parsed_instruction.to_dict(),
        'detection_prompts': detection_prompts,
        'planner_config': planner_config,
        'trials': total,
        'successes': successes,
        'success_rate': successes / total,
        'avg_time': np.mean(successful_times) if successful_times else None,
        'std_time': np.std(successful_times) if successful_times else None,
        'avg_distance': np.mean(successful_dists) if successful_dists else None,
        'std_distance': np.std(successful_dists) if successful_dists else None,
        'plant_blocking_triggered': plant_blocks,
        'headless': args.headless,
        'extra_warmup': args.extra_warmup,
        'tier': args.tier,
    }
    
    with open(os.path.join(batch_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Shutdown
    simulation_app.close()
    
    return successes == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
