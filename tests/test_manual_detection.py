"""
MANUAL ROBOT CONTROL + CONTINUOUS DETECTION

You control the robot manually in Isaac Sim GUI.
This script continuously captures and detects, saving frames to a folder.

Instructions:
1. Run this script
2. In Isaac Sim viewport, select the Turtlebot
3. Use the Move/Rotate tools to position the robot manually
4. Watch the console for detections
5. Check the output folder for saved frames with bounding boxes
"""

import os
import sys
import numpy as np
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("MANUAL CONTROL + CONTINUOUS DETECTION")
    print("="*60)
    print("\nMove the robot manually in Isaac Sim GUI.")
    print("Detections will be saved every 2 seconds.")
    print("Press Ctrl+C to stop.\n")
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from pxr import UsdGeom, Gf
        from PIL import Image, ImageDraw
        
        # Create output folder
        output_dir = os.path.join(project_root, "detection_frames")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output folder: {output_dir}")
        
        # Clear old frames
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        print("Cleared old frames.\n")
        
        print("[1/4] Setting up scene...")
        scene = setup_scene('partial', seed=42)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        
        print("[2/4] Creating camera...")
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
        
        print("[3/4] Loading perception...")
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
        
        # Settle
        print("[4/4] Initializing...")
        for _ in range(50):
            world.step(render=True)
        
        print("\n" + "="*60)
        print("🎮 READY! Move the robot in Isaac Sim GUI.")
        print("="*60)
        print("\nTips:")
        print("  - Select 'Turtlebot' in Stage panel")
        print("  - Use W key for translate tool, E for rotate")
        print("  - Drag the robot to see around the plant")
        print("  - Watch for 🎯 CHARGER detection!\n")
        
        frame_count = 0
        update_interval = 2.0  # seconds between captures
        last_capture = 0
        charger_ever_found = False
        
        while simulation_app.is_running():
            world.step(render=True)
            
            current_time = time.time()
            if current_time - last_capture >= update_interval:
                last_capture = current_time
                frame_count += 1
                
                # Get robot pose
                pose = robot_interface.get_pose()
                
                # Capture RGB
                rgb = robot_interface.get_rgb()
                
                # Detect
                detections = perception.detect(rgb, pose)
                
                # Print status
                print(f"\n--- Frame {frame_count} ---")
                print(f"Robot: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
                
                charger_found = False
                if detections:
                    for det in detections:
                        if det['class'] == 'charger':
                            charger_found = True
                            charger_ever_found = True
                            print(f"  🎯 CHARGER! conf={det['confidence']:.2f}, color={det['color_score']:.2f}, dist={det['estimated_distance']:.1f}m")
                        else:
                            print(f"     {det['class']}: conf={det['confidence']:.2f}")
                else:
                    print("  No detections")
                
                # Analyze green pixels
                green_mask = (rgb[:,:,1] > 80) & (rgb[:,:,1] > rgb[:,:,0] + 20) & (rgb[:,:,1] > rgb[:,:,2] + 20)
                green_pct = np.mean(green_mask) * 100
                if green_pct > 0.5:
                    print(f"  Green pixels: {green_pct:.1f}%")
                
                # Draw bounding boxes
                img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img)
                colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
                
                for det in detections:
                    x1, y1, x2, y2 = det['bbox_2d']
                    color = colors.get(det['class'], (255, 255, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    label_y = y1 - 20 if y1 > 25 else y1 + 5
                    draw.text((x1, label_y), f"{det['class']}: {det['confidence']:.2f}", fill=color)
                
                # Add frame info
                draw.rectangle([0, 0, 200, 60], fill=(0, 0, 0))
                draw.text((5, 5), f"Frame {frame_count}", fill=(255, 255, 255))
                draw.text((5, 20), f"Pos: ({pose[0]:.1f}, {pose[1]:.1f})", fill=(255, 255, 255))
                draw.text((5, 35), f"Yaw: {np.degrees(pose[2]):.0f}°", fill=(255, 255, 255))
                
                if charger_found:
                    draw.rectangle([0, 50, 200, 70], fill=(0, 150, 0))
                    draw.text((5, 52), "🎯 CHARGER FOUND!", fill=(255, 255, 255))
                
                # Save frame
                filename = f"frame_{frame_count:04d}.png"
                if charger_found:
                    filename = f"frame_{frame_count:04d}_CHARGER.png"
                save_path = os.path.join(output_dir, filename)
                img.save(save_path)
        
        print(f"\n\nTotal frames captured: {frame_count}")
        if charger_ever_found:
            print("🎉 Charger was detected at least once!")
        else:
            print("⚠️ Charger was never detected.")
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        print(f"Frames saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
