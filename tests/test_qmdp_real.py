"""
QMDP Navigation Demo - TRUE BELIEF-BASED PLANNING

This script uses the REAL QMDPPlannerReal instead of the rule-based planner.
The key difference is that action selection is based on Q-value computation
over the belief state, not if-else rules.

To compare with original: run test_pomdp_navigation_smooth.py
"""

import os
import sys
import numpy as np
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the original test script and override the planner import
from test_pomdp_navigation_smooth import (
    is_path_blocked_by_plant,
    SmoothMotionController,
)


def main():
    print("="*60)
    print("TRUE QMDP NAVIGATION - BELIEF-BASED PLANNING")
    print("="*60)
    print("Using QMDPPlannerReal instead of rule-based planner")
    print("="*60)
    
    output_dir = os.path.join(project_root, "qmdp_demo_real")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean old outputs
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(output_dir, f))
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from PIL import Image, ImageDraw
        
        print("\n[1/7] Setting up scene...")
        scene = setup_scene('partial', seed=42, use_apriltag=True)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        camera_path = scene['camera_path']
        
        print("\n[2/7] Using camera from scene setup...")
        print(f"  Camera path: {camera_path}")
        
        robot_interface = IsaacSimRobot(
            robot=robot, world=world, 
            camera_path=camera_path, objects=objects
        )
        
        print("\n[3/7] Creating smooth motion controller...")
        motion_controller = SmoothMotionController(robot, world)
        
        print("\n[4/7] Loading HYBRID perception (AprilTag + Grounding DINO)...")
        from perception.apriltag_detector import HybridPerceptionModule, AprilTagChargerDetector
        from perception.vision_model import PerceptionModule, create_isaac_raycast_func
        from omni.physx import get_physx_scene_query_interface
        
        actual_hfov_deg = 2 * np.degrees(np.arctan(20.955 / (2 * 24.0)))
        print(f"  Camera HFOV: {actual_hfov_deg:.1f}°")
        
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
        print("  ✅ Hybrid perception: AprilTag + Grounding DINO")
        
        print("\n[5/7] Creating semantic grid...")
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
        
        print("\n[6/7] Creating TRUE QMDP planner (not rule-based!)...")
        # THIS IS THE KEY DIFFERENCE - use QMDPPlannerReal
        from planner.qmdp_planner_real import QMDPPlannerReal
        
        planner = QMDPPlannerReal(
            grid=grid,
            # Reward model parameters
            reward_dock=100.0,
            reward_step=-1.0,
            lambda_risk=0.5,
            lambda_ig=2.0,
            # Confidence thresholds
            high_conf_threshold=0.7,
            medium_conf_threshold=0.4,
            low_conf_threshold=0.15,
            # Room bounds
            room_bounds=(-2.7, -2.7, 2.7, 2.7),
            obstacle_avoidance_radius=0.25,
        )
        print("  ✅ QMDPPlannerReal - belief-based action selection")
        
        # Initialize simulation
        world.reset()
        print("\n[7/7] Warming up simulation (150 steps)...")
        for _ in range(150):
            world.step(render=True)
        
        robot_interface.refresh_viewport()
        
        # Setup raycast depth sensor
        print("  Setting up raycast depth sensor...")
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
        
        def save_visualization(step, mode, robot_pose, detections, target=None, rgb_frame=None):
            """Save combined visualization with Q-values."""
            vis = grid.visualize(robot_pose)
            grid_img = Image.fromarray(vis)
            grid_img = grid_img.resize((400, 400), Image.NEAREST)
            
            if rgb_frame is None:
                rgb_frame = robot_interface.get_rgb()
            cam_img = Image.fromarray(rgb_frame)
            cam_img = cam_img.resize((400, 300), Image.LANCZOS)
            
            cam_draw = ImageDraw.Draw(cam_img)
            colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
            for det in detections:
                x1, y1, x2, y2 = det['bbox_2d']
                x1 = int(x1 * 400 / 640)
                x2 = int(x2 * 400 / 640)
                y1 = int(y1 * 300 / 480)
                y2 = int(y2 * 300 / 480)
                color = colors.get(det['class'], (255, 255, 0))
                cam_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                depth_str = f"{det['estimated_distance']:.1f}m"
                cam_draw.text((x1, max(5, y1-12)), f"{det['class']} {depth_str}", fill=color)
            
            combined = Image.new('RGB', (820, 480), (255, 255, 255))
            combined.paste(grid_img, (10, 10))
            combined.paste(cam_img, (420, 10))
            
            draw = ImageDraw.Draw(combined)
            
            charger_pos, charger_conf = grid.get_charger_estimate()
            plant_pos, plant_conf = grid.get_class_estimate('plant')
            
            draw.rectangle([420, 320, 810, 470], fill=(240, 240, 240))
            draw.text((430, 325), f"Step {step} | QMDP: {mode}", fill=(0, 0, 0))
            draw.text((430, 345), f"Robot: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f})", fill=(0, 0, 0))
            
            draw.text((430, 365), f"Confidence: {charger_conf:.2f}", fill=(0, 0, 0))
            bar_width = int(150 * min(1.0, charger_conf))
            draw.rectangle([540, 367, 540 + 150, 377], outline=(100, 100, 100))
            if bar_width > 0:
                bar_color = (0, 200, 0) if charger_conf > 0.4 else (255, 165, 0)
                draw.rectangle([540, 367, 540 + bar_width, 377], fill=bar_color)
            
            if plant_pos:
                draw.text((430, 385), f"Plant: ({plant_pos[0]:.1f}, {plant_pos[1]:.1f})", fill=(200, 0, 0))
            
            if charger_pos:
                draw.text((430, 405), f"Charger: ({charger_pos[0]:.1f}, {charger_pos[1]:.1f})", fill=(0, 150, 0))
            
            if target:
                draw.text((430, 440), f"Target: ({target[0]:.1f}, {target[1]:.1f})", fill=(0, 0, 150))
            
            draw.text((10, 460), "TRUE QMDP - Belief-based action selection", fill=(0, 0, 128))
            
            save_path = os.path.join(output_dir, f"step_{step:02d}_{mode}.png")
            combined.save(save_path)
            return save_path
        
        print("\n" + "="*60)
        print("STARTING TRUE QMDP NAVIGATION")
        print("="*60)
        
        navigation_start_time = time.time()
        
        # Initial observation
        print("\n" + "="*50)
        print("INITIAL OBSERVATION (Step 0)")
        print("="*50)
        
        print("  Stabilizing camera (100 steps)...")
        for _ in range(100):
            world.step(render=True)
        
        pose = robot_interface.get_pose()
        print(f"Robot at: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
        
        cam_pos = get_camera_world_pos(pose)
        perception.update_camera_pose(cam_pos, pose[2])
        
        rgb = robot_interface.get_rgb()
        img_mean = np.mean(rgb)
        
        if img_mean >= 100:
            detections = perception.detect(rgb, pose)
        else:
            detections = []
        
        print(f"Initial detections: {len(detections)}")
        for det in detections:
            print(f"  {det['class']}: conf={det['confidence']:.2f}, dist={det['estimated_distance']:.2f}m")
        
        grid.update_from_detections(detections, pose)
        save_visualization(0, "INITIAL", pose, detections, None, rgb)
        
        print("\n" + "="*60)
        print("QMDP NAVIGATION LOOP")
        print("="*60)
        
        max_steps = 25
        step = 0
        success = False
        total_sim_steps = 0
        
        while step < max_steps:
            step += 1
            
            print(f"\n{'='*50}")
            print(f"STEP {step}")
            print(f"{'='*50}")
            
            motion_controller.stop()
            for _ in range(50):
                world.step(render=True)
            
            pose = robot_interface.get_pose()
            print(f"Robot at: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
            
            cam_pos = get_camera_world_pos(pose)
            perception.update_camera_pose(cam_pos, pose[2])
            
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            print(f"Detections: {len(detections)}")
            for det in detections:
                print(f"  {det['class']}: conf={det['confidence']:.2f}, dist={det['estimated_distance']:.2f}m")
            
            grid.update_from_detections(detections, pose)
            
            # QMDP action selection
            target, mode = planner.get_navigation_target(pose)
            
            print(f"\nQMDP Selected: {mode}")
            
            save_visualization(step, mode, pose, detections, target, rgb)
            
            # Check termination
            if mode == "DOCK":
                print("\n🎉 DOCKING based on QMDP decision!")
                
                if target:
                    old_tolerance = motion_controller.POSITION_TOLERANCE
                    motion_controller.POSITION_TOLERANCE = 0.08
                    
                    reached, steps_taken = motion_controller.move_to(
                        target[0], target[1], target[2],
                        max_steps=500
                    )
                    total_sim_steps += steps_taken
                    
                    motion_controller.POSITION_TOLERANCE = old_tolerance
                
                success = True
                save_visualization(step, "DOCK_COMPLETE", motion_controller.get_pose(), detections, target)
                break
            
            # Execute action
            if mode == "FORCED_BACKUP":
                backed_up, steps_taken = motion_controller.backup(distance=0.4, max_steps=400)
                total_sim_steps += steps_taken
                continue
            elif mode in ["ROTATE", "RESCAN"]:
                if target:
                    motion_controller._rotate_to(target[2], max_steps=300)
                    total_sim_steps += 300
                continue
            elif target:
                reached, steps_taken = motion_controller.move_to(
                    target[0], target[1], target[2],
                    max_steps=1000
                )
                total_sim_steps += steps_taken
        
        # Results
        navigation_end_time = time.time()
        time_to_dock = navigation_end_time - navigation_start_time
        
        print("\n" + "="*60)
        print("FINAL RESULTS - TRUE QMDP")
        print("="*60)
        
        final_pose = robot_interface.get_pose()
        gt = robot_interface.get_object_ground_truth()
        actual_charger = gt.get('charger', (2.8, 0.0))
        
        ROBOT_RADIUS = 0.07
        CHARGER_HALF_WIDTH = 0.15
        center_dist = np.sqrt((final_pose[0] - actual_charger[0])**2 + (final_pose[1] - actual_charger[1])**2)
        edge_to_face_dist = max(0, center_dist - ROBOT_RADIUS - CHARGER_HALF_WIDTH)
        
        DOCK_SUCCESS_THRESHOLD = 0.10
        docking_success = success and edge_to_face_dist < DOCK_SUCCESS_THRESHOLD
        
        print(f"\n🤖 Robot final: ({final_pose[0]:.2f}, {final_pose[1]:.2f})")
        print(f"📏 Distance (edge-to-face): {edge_to_face_dist:.2f}m")
        print(f"⏱️ Time to dock: {time_to_dock:.1f}s")
        
        if docking_success:
            print("\n🎉 MISSION SUCCESS - TRUE QMDP DOCKED!")
        else:
            print("\n❌ Mission incomplete")
        
        print(f"\n📊 QMDP Planner Stats:")
        status = planner.get_status()
        for k, v in status.items():
            print(f"   {k}: {v}")
        
        print(f"\n💾 Demo saved to: {output_dir}/")
        
        print("\nKeep window open - Ctrl+C to exit...")
        while simulation_app.is_running():
            world.step(render=True)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
