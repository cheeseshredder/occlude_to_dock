"""
Quick test: Can we detect the green charger at all?
Start robot at a position where charger is visible.
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("GREEN CHARGER DETECTION TEST")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.types import ArticulationAction
        from PIL import Image, ImageDraw
        
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
                'charger': 'green box . green cube',
                'plant': 'red box . red cube',
                'obstacle': 'gray box . gray cube',
            },
            temperature=2.0,
            threshold=0.25,  # Lower threshold
        )
        
        # Settle
        for _ in range(50):
            world.step(render=True)
        
        print("\n[4/4] Testing detection from multiple angles...")
        
        def move_forward(steps=40):
            for _ in range(steps):
                robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([5.0, 5.0])))
                world.step(render=True)
            robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
            for _ in range(10):
                world.step(render=True)
        
        def turn(direction, steps=60):
            vel = np.array([-3.0, 3.0]) if direction == 'left' else np.array([3.0, -3.0])
            for _ in range(steps):
                robot.apply_wheel_actions(ArticulationAction(joint_velocities=vel))
                world.step(render=True)
            robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
            for _ in range(10):
                world.step(render=True)
        
        def test_detection(label):
            pose = robot_interface.get_pose()
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            print(f"\n--- {label} ---")
            print(f"Robot: ({pose[0]:.2f}, {pose[1]:.2f}, {np.degrees(pose[2]):.1f}°)")
            
            charger_found = False
            for det in detections:
                print(f"  {det['class']}: conf={det['confidence']:.2f}, color={det['color_score']:.2f}")
                if det['class'] == 'charger':
                    charger_found = True
            
            if not detections:
                print("  No detections")
            
            # Save image with bboxes
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
            for det in detections:
                x1, y1, x2, y2 = det['bbox_2d']
                color = colors.get(det['class'], (255, 255, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, max(5, y1-15)), f"{det['class']}: {det['confidence']:.2f}", fill=color)
            
            save_path = os.path.join(project_root, f"charger_test_{label.replace(' ', '_')}.png")
            img.save(save_path)
            print(f"  Saved: {save_path}")
            
            return charger_found
        
        # Test 1: From start (charger occluded by plant)
        test_detection("1_start")
        
        # Test 2: Move far left to see around plant
        print("\n>>> Moving LEFT to see around plant...")
        turn('left', 70)
        move_forward(80)
        turn('right', 70)
        test_detection("2_left_side")
        
        # Test 3: Move forward to get closer
        move_forward(60)
        test_detection("3_left_closer")
        
        # Test 4: Turn more right to face charger
        turn('right', 40)
        test_detection("4_facing_charger")
        
        # Test 5: Move right side instead
        print("\n>>> Trying RIGHT side...")
        turn('right', 120)
        move_forward(100)
        turn('left', 60)
        test_detection("5_right_side")
        
        # Test 6: Get even further right
        turn('right', 30)
        move_forward(50)
        turn('left', 30)
        test_detection("6_far_right")
        
        print("\n" + "="*60)
        print("Ground Truth:")
        gt = robot_interface.get_object_ground_truth()
        for name, pos in gt.items():
            print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
        print("="*60)
        
        print("\nCheck the saved images to see what the robot sees!")
        print("Look for GREEN bounding boxes = charger detected")
        
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
