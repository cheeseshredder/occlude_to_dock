"""
POMDP Navigation Demo with REAL-TIME HEATMAP VISUALIZATION

This is a modified version of test_pomdp_navigation_smooth.py that adds
a live-updating belief heatmap window for presentations and demos.

Features:
- Real-time belief map visualization (12x12 grid)
- Camera feed with detection overlays
- Status bar with action, confidence, timing
- All the existing navigation functionality

Usage:
    python test_pomdp_demo.py [--tier partial|heavy] [--seed 42]
"""

import os
import sys
import numpy as np
import time
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def is_path_blocked_by_plant(robot_pos, charger_pos, plant_pos, clearance=0.4):
    """Check if plant blocks the straight-line path from robot to charger."""
    rx, ry = robot_pos
    cx, cy = charger_pos
    px, py = plant_pos
    
    dx = cx - rx
    dy = cy - ry
    path_length = np.sqrt(dx**2 + dy**2)
    
    if path_length < 0.01:
        return False
    
    dx_norm = dx / path_length
    dy_norm = dy / path_length
    
    px_rel = px - rx
    py_rel = py - ry
    
    projection = px_rel * dx_norm + py_rel * dy_norm
    
    if projection < 0 or projection > path_length:
        return False
    
    perp_dist = abs(px_rel * dy_norm - py_rel * dx_norm)
    return perp_dist < clearance


class SmoothMotionController:
    """Smooth motion controller using velocity commands."""
    
    WHEEL_RADIUS = 0.033
    WHEEL_BASE = 0.16
    MAX_LINEAR_VEL = 0.14  # Balanced speed for demo
    MAX_ANGULAR_VEL = 0.7  # Moderate turning speed
    KP_LINEAR = 0.6  # Smoother acceleration
    KP_ANGULAR = 1.3  # Moderate turning response
    POSITION_TOLERANCE = 0.15
    ANGLE_TOLERANCE = 0.1
    
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world
        from omni.isaac.core.utils.types import ArticulationAction
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
        
    def move_to(self, target_x, target_y, target_yaw=None, max_steps=500, viz_callback=None):
        """Move to target with optional visualization callback."""
        print(f"  🚗 Moving to ({target_x:.2f}, {target_y:.2f})...")
        
        position_history = []
        HISTORY_WINDOW = 150  # Longer window for less false positives
        MIN_MOVEMENT_IN_WINDOW = 0.03  # More lenient threshold
        stuck_check_interval = 75  # Check less frequently
        
        for step in range(max_steps):
            x, y, yaw = self.get_pose()
            position_history.append((x, y, step))
            if len(position_history) > HISTORY_WINDOW:
                position_history.pop(0)
            
            dx = target_x - x
            dy = target_y - y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Stuck detection
            if step > 0 and step % stuck_check_interval == 0 and len(position_history) >= HISTORY_WINDOW:
                oldest = position_history[0]
                movement_in_window = np.sqrt((x - oldest[0])**2 + (y - oldest[1])**2)
                if movement_in_window < MIN_MOVEMENT_IN_WINDOW:
                    self.stop()
                    for _ in range(30):
                        self.world.step(render=True)
                    print(f"  ⚠️ STUCK at ({x:.2f}, {y:.2f})")
                    return False, step
            
            target_heading = np.arctan2(dy, dx)
            heading_error = self.normalize_angle(target_heading - yaw)
            
            if distance < self.POSITION_TOLERANCE:
                self.stop()
                if target_yaw is not None:
                    self._rotate_to(target_yaw, max_steps=100)
                for _ in range(30):
                    self.world.step(render=True)
                print(f"  ✅ Reached target in {step} steps")
                return True, step
            
            if abs(heading_error) > 0.3:
                v_linear = 0.0
                v_angular = self.KP_ANGULAR * heading_error
            else:
                v_linear = self.KP_LINEAR * min(distance, 0.5)
                v_angular = self.KP_ANGULAR * heading_error
            
            self.set_wheel_velocities(v_linear, v_angular)
            self.world.step(render=True)
            
            # Call visualization callback periodically
            if viz_callback and step % 5 == 0:
                viz_callback()
        
        self.stop()
        for _ in range(30):
            self.world.step(render=True)
        print(f"  ⚠️ Did not reach target in {max_steps} steps")
        return False, max_steps
    
    def _rotate_to(self, target_yaw, max_steps=200, viz_callback=None):
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
            
            if viz_callback and step % 5 == 0:
                viz_callback()
        
        self.stop()
        for _ in range(30):
            self.world.step(render=True)
        return False
    
    def backup(self, distance=0.3, max_steps=300, viz_callback=None):
        print(f"  ⏪ Backing up {distance:.2f}m...")
        start_x, start_y, _ = self.get_pose()
        position_history = []
        HISTORY_WINDOW = 50
        MIN_MOVEMENT_IN_WINDOW = 0.02
        
        for step in range(max_steps):
            x, y, yaw = self.get_pose()
            position_history.append((x, y))
            if len(position_history) > HISTORY_WINDOW:
                position_history.pop(0)
            
            dist_moved = np.sqrt((x - start_x)**2 + (y - start_y)**2)
            if dist_moved >= distance:
                self.stop()
                for _ in range(30):
                    self.world.step(render=True)
                print(f"  ✅ Backed up {dist_moved:.2f}m")
                return True, step
            
            if len(position_history) >= HISTORY_WINDOW and step % 25 == 0:
                oldest = position_history[0]
                movement_in_window = np.sqrt((x - oldest[0])**2 + (y - oldest[1])**2)
                if movement_in_window < MIN_MOVEMENT_IN_WINDOW:
                    self.stop()
                    for _ in range(30):
                        self.world.step(render=True)
                    print(f"  ⚠️ Backup stuck")
                    return dist_moved > 0.05, step
            
            self.set_wheel_velocities(-self.MAX_LINEAR_VEL, 0.0)
            self.world.step(render=True)
            
            if viz_callback and step % 5 == 0:
                viz_callback()
        
        self.stop()
        for _ in range(30):
            self.world.step(render=True)
        dist_moved = np.sqrt((self.get_pose()[0] - start_x)**2 + (self.get_pose()[1] - start_y)**2)
        return dist_moved > 0.05, max_steps


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='POMDP Navigation Demo with Real-time Visualization')
    parser.add_argument('--tier', type=str, default='partial', choices=['partial', 'heavy'],
                       help='Occlusion tier (partial or heavy)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("="*70)
    print("  OCCLUDE-TO-DOCK: REAL-TIME VISUALIZATION DEMO")
    print("="*70)
    print(f"  Tier: {args.tier} | Seed: {args.seed}")
    print("="*70)
    
    output_dir = os.path.join(project_root, "pomdp_demo_realtime")
    os.makedirs(output_dir, exist_ok=True)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    # Real-time visualizer (initialized later after grid is created)
    realtime_viz = None
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from PIL import Image, ImageDraw
        
        print("\n[1/8] Setting up scene...")
        scene = setup_scene(args.tier, seed=args.seed, use_apriltag=True)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        camera_path = scene['camera_path']
        
        print("\n[2/8] Setting up robot interface...")
        robot_interface = IsaacSimRobot(
            robot=robot, world=world, 
            camera_path=camera_path, objects=objects
        )
        
        print("\n[3/8] Creating motion controller...")
        motion_controller = SmoothMotionController(robot, world)
        
        print("\n[4/8] Loading perception modules...")
        from perception.apriltag_detector import HybridPerceptionModule, AprilTagChargerDetector
        from perception.vision_model import PerceptionModule, create_isaac_raycast_func
        from omni.physx import get_physx_scene_query_interface
        
        actual_hfov_deg = 2 * np.degrees(np.arctan(20.955 / (2 * 24.0)))
        
        grounding_dino = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompts={
                'charger': 'green box . green cube . charging station',
                'plant': 'red box . red cube . red object',
                'obstacle': 'gray box . gray cube',
            },
            platt_a=1.509,
            platt_b=1.598,
            threshold=0.2,
            use_depth_sensor=True,
            depth_noise_enabled=True,
            hfov_deg=actual_hfov_deg,
            img_width=640,
            img_height=480,
        )
        
        apriltag_detector = AprilTagChargerDetector(
            charger_tag_id=0,
            tag_size=0.15,
            hfov_deg=None,
            focal_length_mm=24.0,
            horizontal_aperture_mm=20.955,
            img_width=640,
            img_height=480,
        )
        
        perception = HybridPerceptionModule(
            grounding_dino_detector=grounding_dino,
            apriltag_detector=apriltag_detector,
        )
        
        print("\n[5/8] Creating semantic grid...")
        from mapping.semantic_grid import SemanticGrid
        
        grid = SemanticGrid(
            size_meters=(6.0, 6.0),
            resolution=0.5,
            origin=(-3.0, -3.0),
            close_range_threshold=1.2,
            close_range_weight_factor=0.2,
            lock_threshold=1.0,
            min_detections_to_lock=2,
        )
        
        print("\n[6/8] Creating POMDP planner...")
        from planner.pomdp_planner import POMDPPlanner
        
        planner = POMDPPlanner(
            grid=grid,
            num_candidates=12,
            confidence_threshold=0.4,
            obstacle_avoidance_radius=0.25,
            room_bounds=(-2.7, -2.7, 2.7, 2.7),
        )
        
        print("\n[7/8] Creating REAL-TIME VISUALIZER...")
        from visualization.realtime_heatmap import RealtimeVisualizer
        
        viz_output_dir = os.path.join(project_root, "demo_realtime_frames")
        realtime_viz = RealtimeVisualizer(
            grid=grid,
            window_name=f"Occlude-to-Dock Demo | {args.tier.upper()} | Seed {args.seed}",
            show_camera=True,
            output_dir=viz_output_dir,
            cell_size=40,
            camera_size=(400, 300),
            save_all_frames=True,  # Save all frames for video creation
        )
        
        print(f"\n" + "="*70)
        print(f"  REAL-TIME VISUALIZATION")
        print(f"="*70)
        print(f"  Frames are saved to: {viz_output_dir}/")
        print(f"  Open 'current.png' in an image viewer for live updates.")
        print(f"  TIP: Use Windows Photo Viewer or IrfanView with auto-refresh.")
        print(f"="*70)
        
        # Initialize simulation
        world.reset()
        print("  Warming up simulation...")
        for _ in range(150):
            world.step(render=True)
        
        robot_interface.refresh_viewport()
        
        print("\n[8/8] Setting up raycast depth...")
        physx_query = get_physx_scene_query_interface()
        raycast_func = create_isaac_raycast_func(physx_query)
        
        CAMERA_OFFSET_X = 0.05
        CAMERA_OFFSET_Y = 0.0
        CAMERA_HEIGHT = 0.20
        
        def get_camera_world_pos(robot_pose):
            rx, ry, ryaw = robot_pose
            cos_yaw = np.cos(ryaw)
            sin_yaw = np.sin(ryaw)
            cam_x = rx + CAMERA_OFFSET_X * cos_yaw - CAMERA_OFFSET_Y * sin_yaw
            cam_y = ry + CAMERA_OFFSET_X * sin_yaw + CAMERA_OFFSET_Y * cos_yaw
            cam_z = CAMERA_HEIGHT
            return np.array([cam_x, cam_y, cam_z])
        
        initial_pose = robot_interface.get_pose()
        initial_cam_pos = get_camera_world_pos(initial_pose)
        perception.set_raycast_func(raycast_func, initial_cam_pos)
        
        # ============================================================
        # MAIN NAVIGATION LOOP WITH REAL-TIME VISUALIZATION
        # ============================================================
        
        print("\n" + "="*70)
        print("  STARTING NAVIGATION WITH REAL-TIME VISUALIZATION")
        print("="*70)
        print("  Press 'q' in the visualization window to stop")
        print("="*70)
        
        navigation_start_time = time.time()
        
        # Current state for visualization updates
        current_rgb = None
        current_detections = []
        current_action = "INITIALIZING"
        current_step = 0
        
        def update_visualization():
            """Update the real-time visualization (saves frame to disk)."""
            nonlocal realtime_viz
            if realtime_viz is None:
                return
            
            pose = motion_controller.get_pose()
            realtime_viz.update(
                robot_pose=pose,
                detections=current_detections,
                rgb_frame=current_rgb,
                action=current_action,
                info={'step': current_step}
            )
        
        # Initial observation
        print("\n--- INITIAL OBSERVATION ---")
        for _ in range(100):
            world.step(render=True)
        
        pose = robot_interface.get_pose()
        cam_pos = get_camera_world_pos(pose)
        perception.update_camera_pose(cam_pos, pose[2])
        
        current_rgb = robot_interface.get_rgb()
        img_mean = np.mean(current_rgb)
        img_std = np.std(current_rgb)
        
        if img_mean >= 100 and img_std >= 20:
            current_detections = perception.detect(current_rgb, pose)
        else:
            current_detections = []
        
        grid.update_from_detections(current_detections, pose)
        
        current_action = "INITIAL"
        update_visualization()
        
        # Navigation loop
        max_steps = 25
        step = 0
        success = False
        total_sim_steps = 0
        dock_failure_count = 0
        
        while step < max_steps:
            step += 1
            current_step = step
            
            print(f"\n{'='*50}")
            print(f"STEP {step}")
            print(f"{'='*50}")
            
            # Stabilize (keep previous detections for smooth viz)
            motion_controller.stop()
            for i in range(50):
                world.step(render=True)
                # Only update viz at end of stabilization, not during
            
            # Now clear old detections and get fresh ones
            current_detections = []
            
            pose = motion_controller.get_pose()
            print(f"Robot at: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
            
            cam_pos = get_camera_world_pos(pose)
            perception.update_camera_pose(cam_pos, pose[2])
            
            current_rgb = robot_interface.get_rgb()
            current_detections = perception.detect(current_rgb, pose)
            
            print(f"Detections: {len(current_detections)}")
            for det in current_detections:
                marker = "🎯" if det['class'] == 'charger' else "🌱" if det['class'] == 'plant' else "  "
                print(f"  {marker} {det['class']}: {det['estimated_distance']:.2f}m")
            
            grid.update_from_detections(current_detections, pose)
            
            # Update visualization with fresh detections
            update_visualization()
            
            # ============================================================
            # VISUAL DOCK TRIGGER - bypass planner when charger clearly visible
            # ============================================================
            charger_detection = None
            for det in current_detections:
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
                    
                    # Check plant blocking using geometric path check
                    plant_blocking = False
                    if grid.locked_estimates.get('plant') is not None:
                        px, py = grid.locked_estimates['plant']['position']
                        plant_blocking = is_path_blocked_by_plant(
                            robot_pos=(rx, ry),
                            charger_pos=(cx, cy),
                            plant_pos=(px, py),
                            clearance=0.4
                        )
                        if plant_blocking:
                            print(f"  🌱 Plant at ({px:.2f}, {py:.2f}) blocking path!")
                        else:
                            print(f"  ✅ Path clear to charger")
                    
                    if not plant_blocking:
                        print(f"  🎯 VISUAL DOCK TRIGGER: charger at {visual_dist:.2f}m")
                        current_action = "VISUAL_DOCK"
                        update_visualization()
                        
                        # Attempt dock
                        approach_dist = max(0.1, visual_dist - 0.1)
                        target_x = rx + approach_dist * np.cos(ryaw)
                        target_y = ry + approach_dist * np.sin(ryaw)
                        
                        old_tolerance = motion_controller.POSITION_TOLERANCE
                        motion_controller.POSITION_TOLERANCE = 0.08
                        
                        dock_viz_counter = [0]  # Use list to allow modification in nested function
                        def dock_viz_callback():
                            nonlocal current_rgb, current_detections
                            dock_viz_counter[0] += 1
                            pose = motion_controller.get_pose()
                            current_rgb = robot_interface.get_rgb()
                            # Run fresh detection every 20 viz updates for live boxes
                            if dock_viz_counter[0] % 20 == 0:
                                cam_pos = get_camera_world_pos(pose)
                                perception.update_camera_pose(cam_pos, pose[2])
                                current_detections = perception.detect(current_rgb, pose)
                            realtime_viz.update(
                                robot_pose=pose,
                                detections=current_detections,
                                rgb_frame=current_rgb,
                                action="VISUAL_DOCK",
                                info={'step': current_step}
                            )
                        
                        reached, steps_taken = motion_controller.move_to(
                            target_x, target_y, max_steps=400,
                            viz_callback=dock_viz_callback
                        )
                        total_sim_steps += steps_taken
                        motion_controller.POSITION_TOLERANCE = old_tolerance
                        
                        if not reached:
                            dock_failure_count += 1
                            print(f"  ⚠️ Dock attempt failed ({dock_failure_count}/3)")
                            if dock_failure_count >= 3:
                                print("  ❌ Too many dock failures")
                                break
                            motion_controller.backup(0.3)
                            continue
                        
                        # Check final distance
                        final_pose = motion_controller.get_pose()
                        final_dist = np.sqrt((final_pose[0] - 2.8)**2 + (final_pose[1] - 0.0)**2)
                        
                        if final_dist < 0.30:
                            success = True
                            current_action = "DOCK_SUCCESS"
                            # Get fresh frame and detection for final visualization
                            current_rgb = robot_interface.get_rgb()
                            current_detections = perception.detect(current_rgb, final_pose)
                            update_visualization()
                            print(f"  ✅ DOCKED! Distance: {final_dist:.2f}m")
                            break
            
            # ============================================================
            # PLANNER-BASED NAVIGATION (if visual trigger didn't fire)
            # ============================================================
            if success:
                break
                
            # Get planner target
            target, mode = planner.get_navigation_target(pose)
            current_action = mode
            
            print(f"Planner: {mode}")
            
            # Update visualization with new action
            update_visualization()
            
            # Check termination
            if mode == "DOCK":
                print("\n🎉 DOCKING!")
                current_action = "DOCKING"
                
                if target:
                    old_tolerance = motion_controller.POSITION_TOLERANCE
                    motion_controller.POSITION_TOLERANCE = 0.08
                    
                    planner_dock_counter = [0]
                    def motion_viz_callback():
                        nonlocal current_rgb, current_detections
                        planner_dock_counter[0] += 1
                        pose = motion_controller.get_pose()
                        current_rgb = robot_interface.get_rgb()
                        # Run fresh detection every 20 viz updates for live boxes
                        if planner_dock_counter[0] % 20 == 0:
                            cam_pos = get_camera_world_pos(pose)
                            perception.update_camera_pose(cam_pos, pose[2])
                            current_detections = perception.detect(current_rgb, pose)
                        realtime_viz.update(
                            robot_pose=pose,
                            detections=current_detections,
                            rgb_frame=current_rgb,
                            action="DOCKING",
                            info={'step': current_step}
                        )
                    
                    reached, steps_taken = motion_controller.move_to(
                        target[0], target[1], target[2],
                        max_steps=500,
                        viz_callback=motion_viz_callback
                    )
                    total_sim_steps += steps_taken
                    motion_controller.POSITION_TOLERANCE = old_tolerance
                
                success = True
                current_action = "DOCK_SUCCESS"
                # Get fresh detection for final frame
                current_rgb = robot_interface.get_rgb()
                current_detections = perception.detect(current_rgb, motion_controller.get_pose())
                update_visualization()
                break
            
            # Handle different modes
            if mode == "FORCED_BACKUP":
                current_action = "BACKUP"
                update_visualization()
                
                backup_counter = [0]
                def backup_viz_callback():
                    nonlocal current_rgb, current_detections
                    backup_counter[0] += 1
                    pose = motion_controller.get_pose()
                    current_rgb = robot_interface.get_rgb()
                    # Run fresh detection every 20 viz updates
                    if backup_counter[0] % 20 == 0:
                        cam_pos = get_camera_world_pos(pose)
                        perception.update_camera_pose(cam_pos, pose[2])
                        current_detections = perception.detect(current_rgb, pose)
                    realtime_viz.update(
                        robot_pose=pose,
                        detections=current_detections,
                        rgb_frame=current_rgb,
                        action="BACKUP",
                        info={'step': current_step}
                    )
                
                backed_up, steps_taken = motion_controller.backup(
                    distance=0.4, max_steps=400,
                    viz_callback=backup_viz_callback
                )
                total_sim_steps += steps_taken
                continue
                
            elif mode == "LOOK_AROUND":
                current_action = "SCANNING"
                update_visualization()
                
                scan_counter = [0]
                def scan_viz_callback():
                    nonlocal current_rgb, current_detections
                    scan_counter[0] += 1
                    pose = motion_controller.get_pose()
                    current_rgb = robot_interface.get_rgb()
                    # Run fresh detection every 20 viz updates
                    if scan_counter[0] % 20 == 0:
                        cam_pos = get_camera_world_pos(pose)
                        perception.update_camera_pose(cam_pos, pose[2])
                        current_detections = perception.detect(current_rgb, pose)
                    realtime_viz.update(
                        robot_pose=pose,
                        detections=current_detections,
                        rgb_frame=current_rgb,
                        action="SCANNING",
                        info={'step': current_step}
                    )
                
                if target:
                    motion_controller._rotate_to(
                        target[2], max_steps=300,
                        viz_callback=scan_viz_callback
                    )
                    total_sim_steps += 300
                continue
                
            elif target:
                current_action = mode
                update_visualization()
                
                nav_counter = [0]
                def nav_viz_callback():
                    nonlocal current_rgb, current_detections
                    nav_counter[0] += 1
                    pose = motion_controller.get_pose()
                    current_rgb = robot_interface.get_rgb()
                    # Run fresh detection every 20 viz updates for live boxes
                    if nav_counter[0] % 20 == 0:
                        cam_pos = get_camera_world_pos(pose)
                        perception.update_camera_pose(cam_pos, pose[2])
                        current_detections = perception.detect(current_rgb, pose)
                    realtime_viz.update(
                        robot_pose=pose,
                        detections=current_detections,
                        rgb_frame=current_rgb,
                        action=mode,
                        info={'step': current_step}
                    )
                
                reached, steps_taken = motion_controller.move_to(
                    target[0], target[1], target[2],
                    max_steps=1000,
                    viz_callback=nav_viz_callback
                )
                total_sim_steps += steps_taken
        
        # ============================================================
        # RESULTS
        # ============================================================
        
        navigation_end_time = time.time()
        time_to_dock = navigation_end_time - navigation_start_time
        
        print("\n" + "="*70)
        print("  FINAL RESULTS")
        print("="*70)
        
        gt = robot_interface.get_object_ground_truth()
        actual_charger = gt.get('charger', (2.8, 0.0))
        
        final_pose = robot_interface.get_pose()
        
        ROBOT_RADIUS = 0.07
        CHARGER_HALF_WIDTH = 0.15
        
        center_dist = np.sqrt((final_pose[0] - actual_charger[0])**2 + (final_pose[1] - actual_charger[1])**2)
        edge_to_face_dist = max(0, center_dist - ROBOT_RADIUS - CHARGER_HALF_WIDTH)
        
        DOCK_SUCCESS_THRESHOLD = 0.10
        docking_success = success and edge_to_face_dist < DOCK_SUCCESS_THRESHOLD
        
        # Get fresh frame and detection for final visualization
        current_rgb = robot_interface.get_rgb()
        cam_pos = get_camera_world_pos(final_pose)
        perception.update_camera_pose(cam_pos, final_pose[2])
        current_detections = perception.detect(current_rgb, final_pose)
        
        current_action = "SUCCESS ✅" if docking_success else "FAILED ❌"
        update_visualization()
        
        print(f"\n  Success: {'YES ✅' if docking_success else 'NO ❌'}")
        print(f"  Time to Dock: {time_to_dock:.1f}s")
        print(f"  Final Distance: {edge_to_face_dist:.2f}m")
        print(f"  Navigation Steps: {step}")
        
        print("\n" + "="*70)
        print("  Demo complete!")
        print(f"  All frames saved to: {viz_output_dir}/")
        print(f"  Total frames: {realtime_viz._frame_number}")
        print("="*70)
        print("\nPress Ctrl+C to exit or close Isaac Sim window...")
        
        # Keep simulation running so user can see final state
        try:
            while simulation_app.is_running():
                world.step(render=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if realtime_viz is not None:
            realtime_viz.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
