"""
Full POMDP Navigation Demo - v3 Fixed

Key fix: Properly transition from APPROACH_AROUND to APPROACH
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("POMDP NAVIGATION - v3 (Loop Fix)")
    print("="*60)
    
    output_dir = os.path.join(project_root, "pomdp_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(output_dir, f))
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from pxr import UsdGeom, Gf
        from scipy.spatial.transform import Rotation
        from PIL import Image, ImageDraw
        
        print("\n[1/5] Setting up scene...")
        scene = setup_scene('partial', seed=42)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        
        print("\n[2/5] Creating camera...")
        stage = world.stage
        camera_path = "/World/Turtlebot/base_link/robot_camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        camera_prim.CreateFocalLengthAttr(24.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.20))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -90.0))
        
        robot_interface = IsaacSimRobot(
            robot=robot, world=world, 
            camera_path=camera_path, objects=objects
        )
        
        print("\n[3/5] Loading perception...")
        from perception.vision_model import PerceptionModule
        
        perception = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompts={
                'charger': 'green box . green cube . green object',
                'plant': 'red box . red cube . red object',
                'obstacle': 'gray box . gray cube',
            },
            temperature=2.0,
            threshold=0.2,
        )
        
        print("\n[4/5] Creating semantic grid...")
        from mapping.semantic_grid import SemanticGrid
        
        grid = SemanticGrid(
            size_meters=(6.0, 6.0),
            resolution=0.5,
            origin=(-3.0, -3.0),
            plant_to_charger_offset=0.8,
            close_range_threshold=1.2,
            close_range_weight_factor=0.2,
            lock_threshold=1.0,
            min_detections_to_lock=2,
        )
        
        print("\n[5/5] Creating POMDP planner v3...")
        from planner.pomdp_planner import POMDPPlanner
        
        planner = POMDPPlanner(
            grid=grid,
            num_candidates=12,
            confidence_threshold=0.4,
            plant_avoidance_radius=0.5,
        )
        
        world.reset()
        for _ in range(30):
            world.step(render=True)
        
        def teleport_robot(x, y, yaw_rad):
            x = np.clip(x, -2.0, 2.8)
            y = np.clip(y, -2.5, 2.5)
            
            r = Rotation.from_euler('z', yaw_rad)
            quat = r.as_quat()
            robot.set_world_pose(
                position=np.array([x, y, 0.0]),
                orientation=np.array([quat[3], quat[0], quat[1], quat[2]])
            )
            for _ in range(25):
                world.step(render=True)
        
        def save_visualization(step, mode, robot_pose, detections, target=None):
            vis = grid.visualize(robot_pose)
            grid_img = Image.fromarray(vis)
            grid_img = grid_img.resize((400, 400), Image.NEAREST)
            
            rgb = robot_interface.get_rgb()
            cam_img = Image.fromarray(rgb)
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
                cam_draw.text((x1, max(5, y1-12)), det['class'], fill=color)
            
            combined = Image.new('RGB', (820, 480), (255, 255, 255))
            combined.paste(grid_img, (10, 10))
            combined.paste(cam_img, (420, 10))
            
            draw = ImageDraw.Draw(combined)
            
            charger_pos, charger_conf = grid.get_charger_estimate()
            plant_pos, plant_conf = grid.get_class_estimate('plant')
            
            charger_locked = grid.locked_estimates['charger'] is not None
            plant_locked = grid.locked_estimates['plant'] is not None
            
            draw.rectangle([420, 320, 810, 470], fill=(240, 240, 240))
            draw.text((430, 325), f"Step {step} | Mode: {mode}", fill=(0, 0, 0))
            draw.text((430, 345), f"Robot: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f})", fill=(0, 0, 0))
            
            draw.text((430, 365), f"Confidence: {charger_conf:.2f}", fill=(0, 0, 0))
            bar_width = int(150 * min(1.0, charger_conf))
            draw.rectangle([540, 367, 540 + 150, 377], outline=(100, 100, 100))
            if bar_width > 0:
                bar_color = (0, 200, 0) if charger_conf > 0.4 else (255, 165, 0)
                draw.rectangle([540, 367, 540 + bar_width, 377], fill=bar_color)
            
            if plant_pos:
                lock_icon = "🔒" if plant_locked else ""
                draw.text((430, 385), f"Plant: ({plant_pos[0]:.1f}, {plant_pos[1]:.1f}) {lock_icon}", fill=(200, 0, 0))
            
            if charger_pos:
                lock_icon = "🔒" if charger_locked else ""
                draw.text((430, 405), f"Charger: ({charger_pos[0]:.1f}, {charger_pos[1]:.1f}) {lock_icon}", fill=(0, 150, 0))
            
            if target:
                draw.text((430, 440), f"Target: ({target[0]:.1f}, {target[1]:.1f})", fill=(0, 0, 150))
            
            draw.rectangle([10, 415, 400, 445], fill=(50, 50, 50))
            draw.rectangle([15, 422, 30, 437], fill=(0, 200, 0))
            draw.text((35, 422), "Charger", fill=(255, 255, 255))
            draw.rectangle([100, 422, 115, 437], fill=(200, 0, 0))
            draw.text((120, 422), "Plant", fill=(255, 255, 255))
            draw.rectangle([175, 422, 190, 437], fill=(100, 100, 100))
            draw.text((195, 422), "Obstacle", fill=(255, 255, 255))
            draw.rectangle([270, 422, 285, 437], fill=(0, 0, 255))
            draw.text((290, 422), "Robot", fill=(255, 255, 255))
            
            save_path = os.path.join(output_dir, f"step_{step:02d}_{mode}.png")
            combined.save(save_path)
            return save_path
        
        print("\n" + "="*60)
        print("STARTING POMDP NAVIGATION")
        print("="*60)
        
        max_steps = 15
        step = 0
        
        teleport_robot(0.0, 0.0, 0.0)
        
        success = False
        
        while step < max_steps:
            step += 1
            
            print(f"\n{'='*50}")
            print(f"STEP {step}")
            print(f"{'='*50}")
            
            pose = robot_interface.get_pose()
            print(f"Robot at: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
            
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            print(f"Detections: {len(detections)}")
            for det in detections:
                marker = "🎯" if det['class'] == 'charger' else "🌱" if det['class'] == 'plant' else "  "
                print(f"  {marker} {det['class']}: conf={det['confidence']:.2f}")
            
            grid.update_from_detections(detections, pose)
            
            charger_pos, charger_conf = grid.get_charger_estimate()
            plant_pos, plant_conf = grid.get_class_estimate('plant')
            charger_locked = grid.locked_estimates['charger'] is not None
            plant_locked = grid.locked_estimates['plant'] is not None
            
            print(f"\nEstimates:")
            if plant_pos:
                lock_str = "🔒" if plant_locked else ""
                print(f"  🌱 Plant: ({plant_pos[0]:.2f}, {plant_pos[1]:.2f}) {lock_str}")
            if charger_pos:
                lock_str = "🔒" if charger_locked else ""
                print(f"  🔋 Charger: ({charger_pos[0]:.2f}, {charger_pos[1]:.2f}) {lock_str}")
                print(f"     Confidence: {charger_conf:.2f}")
            
            target, mode = planner.get_navigation_target(pose)
            
            print(f"\nPlanner: {mode} (went_around: {planner.went_around_plant})")
            
            save_path = save_visualization(step, mode, pose, detections, target)
            print(f"Saved: {save_path}")
            
            # Check termination
            if mode == "DOCK":
                print("\n🎉 DOCKING!")
                success = True
                break
            
            # Handle navigation
            if target:
                print(f"\n→ Moving to ({target[0]:.2f}, {target[1]:.2f})")
                teleport_robot(target[0], target[1], target[2])
                
                # If we just went around, mark it
                if mode == "APPROACH_AROUND":
                    # Check if we reached the waypoint
                    new_pose = robot_interface.get_pose()
                    dist_moved = np.sqrt((new_pose[0] - pose[0])**2 + (new_pose[1] - pose[1])**2)
                    if dist_moved > 0.2:
                        print(f"  ✅ Moved around plant, will proceed directly next step")
                        planner.went_around_plant = True
            else:
                print("⚠ No target")
        
        # Final approach if we haven't docked yet
        if not success and charger_pos and charger_conf > 0.3:
            print("\n📍 Final approach to charger...")
            pose = robot_interface.get_pose()
            yaw = np.arctan2(charger_pos[1] - pose[1], charger_pos[0] - pose[0])
            
            for i in range(5):
                pose = robot_interface.get_pose()
                dist = np.sqrt((pose[0] - charger_pos[0])**2 + (pose[1] - charger_pos[1])**2)
                
                if dist < 0.3:
                    success = True
                    break
                
                move = min(0.5, dist - 0.2)
                new_x = pose[0] + move * np.cos(yaw)
                new_y = pose[1] + move * np.sin(yaw)
                
                step += 1
                teleport_robot(new_x, new_y, yaw)
                
                detections = perception.detect(robot_interface.get_rgb(), pose)
                save_visualization(step, "FINAL_APPROACH", robot_interface.get_pose(), detections)
                print(f"  Final approach {i+1}: ({new_x:.2f}, {new_y:.2f})")
            
            # Final dock position
            step += 1
            teleport_robot(charger_pos[0] - 0.1, charger_pos[1], yaw)
            pose = robot_interface.get_pose()
            detections = perception.detect(robot_interface.get_rgb(), pose)
            save_visualization(step, "DOCK", pose, detections)
            success = True
        
        # ============================================================
        # RESULTS
        # ============================================================
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        grid.print_summary()
        
        charger_pos, charger_conf = grid.get_charger_estimate()
        
        print("\n📍 Ground Truth:")
        gt = robot_interface.get_object_ground_truth()
        for name, pos in gt.items():
            print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        if charger_pos:
            error = np.sqrt((charger_pos[0] - 2.8)**2 + (charger_pos[1] - 0.0)**2)
            print(f"\n📏 Charger Estimate Error: {error:.2f}m")
        
        final_pose = robot_interface.get_pose()
        actual_charger = (2.8, 0.0)
        final_dist = np.sqrt((final_pose[0] - actual_charger[0])**2 + (final_pose[1] - actual_charger[1])**2)
        print(f"\n🤖 Robot final: ({final_pose[0]:.2f}, {final_pose[1]:.2f})")
        print(f"🤖 Distance to actual charger: {final_dist:.2f}m")
        
        if success and final_dist < 0.6:
            print("\n🎉 MISSION SUCCESS!")
        elif success:
            print("\n⚠️ Docked but not at actual charger")
        else:
            print("\n❌ Mission incomplete")
        
        print(f"\n💾 Demo saved to: {output_dir}/")
        print(f"   Total steps: {step}")
        
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
