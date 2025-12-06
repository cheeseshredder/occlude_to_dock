"""
Test charger detection by TELEPORTING robot to the exact position
from the screenshot where charger is partially visible.

From screenshot:
- Translate Y: 1.14044
- Orientation Z: 65.748°
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("TELEPORT TEST - Using exact coordinates from screenshot")
    print("Position: (0, 1.14, 0), Yaw: 65.7°")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from pxr import UsdGeom, Gf
        from PIL import Image, ImageDraw
        from scipy.spatial.transform import Rotation
        
        print("\n[1/4] Setting up scene...")
        scene = setup_scene('partial', seed=42)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        
        # Create camera
        print("\n[2/4] Creating camera...")
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
        
        print("\n[3/4] Loading perception...")
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
        
        # Initialize world
        world.reset()
        
        # Settle physics
        for _ in range(30):
            world.step(render=True)
        
        print("\n[4/4] Teleporting robot to screenshot position...")
        
        # EXACT coordinates from screenshot
        target_x = 0.0
        target_y = 1.14044
        target_z = 0.0
        target_yaw_deg = 65.748
        target_yaw_rad = np.radians(target_yaw_deg)
        
        # Convert yaw to quaternion
        r = Rotation.from_euler('z', target_yaw_deg, degrees=True)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Set robot position and orientation
        robot.set_world_pose(
            position=np.array([target_x, target_y, target_z]),
            orientation=np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
        )
        
        print(f"  Target position: ({target_x}, {target_y}, {target_z})")
        print(f"  Target yaw: {target_yaw_deg}°")
        
        # Let it settle
        for _ in range(50):
            world.step(render=True)
        
        # Verify position
        pose = robot_interface.get_pose()
        print(f"  Actual position: ({pose[0]:.2f}, {pose[1]:.2f})")
        print(f"  Actual yaw: {np.degrees(pose[2]):.1f}°")
        
        # Capture and detect
        print("\n" + "="*60)
        print("DETECTION TEST")
        print("="*60)
        
        rgb = robot_interface.get_rgb()
        detections = perception.detect(rgb, pose)
        
        print(f"\nDetections: {len(detections)}")
        charger_found = False
        for det in detections:
            marker = "🎯 CHARGER!" if det['class'] == 'charger' else "  "
            print(f"  {marker} {det['class']}: conf={det['confidence']:.2f}, color={det['color_score']:.2f}, dist={det['estimated_distance']:.1f}m")
            if det['class'] == 'charger':
                charger_found = True
        
        # Analyze image colors
        print("\n--- Image Color Analysis ---")
        green_mask = (rgb[:,:,1] > 80) & (rgb[:,:,1] > rgb[:,:,0] + 20) & (rgb[:,:,1] > rgb[:,:,2] + 20)
        green_pct = np.mean(green_mask) * 100
        red_mask = (rgb[:,:,0] > 80) & (rgb[:,:,0] > rgb[:,:,1] + 20) & (rgb[:,:,0] > rgb[:,:,2] + 20)
        red_pct = np.mean(red_mask) * 100
        print(f"Green pixels: {green_pct:.2f}%")
        print(f"Red pixels: {red_pct:.2f}%")
        
        # Save image with bounding boxes
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
        for det in detections:
            x1, y1, x2, y2 = det['bbox_2d']
            color = colors.get(det['class'], (255, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_y = y1 - 20 if y1 > 25 else y1 + 5
            draw.text((x1, label_y), f"{det['class']}: {det['confidence']:.2f}", fill=color)
        
        save_path = os.path.join(project_root, "teleport_test.png")
        img.save(save_path)
        print(f"\nSaved: {save_path}")
        
        if charger_found:
            print("\n🎉 SUCCESS! GREEN CHARGER DETECTED!")
        else:
            print("\n⚠️ Charger not detected.")
            print("Check teleport_test.png to see what the robot camera sees.")
        
        print("\n📍 Ground Truth:")
        gt = robot_interface.get_object_ground_truth()
        for name, pos in gt.items():
            print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
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
