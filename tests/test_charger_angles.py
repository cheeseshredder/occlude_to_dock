"""
Test charger detection from angles where it's visible
Based on Isaac Sim screenshot showing partial visibility
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("CHARGER VISIBILITY TEST - From Known Good Angles")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from pxr import UsdGeom, Gf, UsdPhysics
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
                'charger': 'green box . green cube . green rectangular object',
                'plant': 'red box . red cube . red rectangular object',
                'obstacle': 'gray box . gray cube',
            },
            temperature=2.0,
            threshold=0.2,  # Lower threshold to catch more
        )
        
        # Settle physics
        for _ in range(50):
            world.step(render=True)
        
        print("\n[4/4] Testing from multiple positions...")
        
        def test_from_position(label, move_commands):
            """Execute move commands and test detection"""
            print(f"\n{'='*50}")
            print(f"Test: {label}")
            print(f"{'='*50}")
            
            # Execute movement
            for cmd in move_commands:
                if cmd[0] == 'forward':
                    steps = cmd[1]
                    for _ in range(steps):
                        robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([5.0, 5.0])))
                        world.step(render=True)
                elif cmd[0] == 'turn_left':
                    steps = cmd[1]
                    for _ in range(steps):
                        robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([-3.0, 3.0])))
                        world.step(render=True)
                elif cmd[0] == 'turn_right':
                    steps = cmd[1]
                    for _ in range(steps):
                        robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([3.0, -3.0])))
                        world.step(render=True)
                
                # Stop
                robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
                for _ in range(5):
                    world.step(render=True)
            
            # Get pose and capture
            pose = robot_interface.get_pose()
            print(f"Robot position: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
            
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            print(f"Detections: {len(detections)}")
            charger_found = False
            for det in detections:
                marker = "🎯" if det['class'] == 'charger' else "  "
                print(f"  {marker} {det['class']}: conf={det['confidence']:.2f}, color={det['color_score']:.2f}, dist={det['estimated_distance']:.1f}m")
                if det['class'] == 'charger':
                    charger_found = True
            
            # Analyze colors in image
            green_mask = (rgb[:,:,1] > 80) & (rgb[:,:,1] > rgb[:,:,0] + 20) & (rgb[:,:,1] > rgb[:,:,2] + 20)
            green_pct = np.mean(green_mask) * 100
            red_mask = (rgb[:,:,0] > 80) & (rgb[:,:,0] > rgb[:,:,1] + 20) & (rgb[:,:,0] > rgb[:,:,2] + 20)
            red_pct = np.mean(red_mask) * 100
            print(f"Image analysis: {green_pct:.1f}% green, {red_pct:.1f}% red")
            
            # Save image with boxes
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
            for det in detections:
                x1, y1, x2, y2 = det['bbox_2d']
                color = colors.get(det['class'], (255, 255, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, max(5, y1-15)), f"{det['class']}: {det['confidence']:.2f}", fill=color)
            
            save_path = os.path.join(project_root, f"test_{label.replace(' ', '_')}.png")
            img.save(save_path)
            print(f"Saved: {save_path}")
            
            return charger_found
        
        # Test positions based on the screenshot
        # The screenshot shows robot at Y~1.14, rotated ~66 degrees
        # Charger is partially visible on the LEFT side
        
        results = []
        
        # Test 1: Start position (charger occluded)
        results.append(("Start", test_from_position("1_start", [])))
        
        # Test 2: Move LEFT (positive Y) - this is where charger was visible in screenshot
        results.append(("Move left", test_from_position("2_move_left", [
            ('turn_left', 50),   # Turn left ~45°
            ('forward', 80),     # Move forward
        ])))
        
        # Test 3: Turn to face charger area
        results.append(("Face charger", test_from_position("3_face_charger", [
            ('turn_right', 30),  # Turn back toward charger
        ])))
        
        # Test 4: Move more left and forward
        results.append(("More left", test_from_position("4_more_left", [
            ('turn_left', 20),
            ('forward', 60),
        ])))
        
        # Test 5: Turn right toward charger
        results.append(("Right toward charger", test_from_position("5_toward_charger", [
            ('turn_right', 60),
        ])))
        
        # Test 6: Approach
        results.append(("Approach", test_from_position("6_approach", [
            ('forward', 50),
        ])))
        
        # Test 7: Go around to the other side
        # Reset by going far right
        results.append(("Far right", test_from_position("7_far_right", [
            ('turn_right', 100),
            ('forward', 100),
            ('turn_left', 80),
        ])))
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        charger_found_any = False
        for label, found in results:
            status = "✅ CHARGER FOUND" if found else "❌ No charger"
            print(f"  {label}: {status}")
            if found:
                charger_found_any = True
        
        if charger_found_any:
            print("\n🎉 SUCCESS! Charger detection is working!")
        else:
            print("\n⚠️ Charger never detected. Check:")
            print("   1. Green color detection threshold")
            print("   2. Grounding DINO prompts")
            print("   3. Charger visibility in saved images")
        
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
