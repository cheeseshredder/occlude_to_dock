"""
Test RGB-only Perception Pipeline with Bounding Box Visualization
Now detects: charger (green), plant (red), obstacles (gray)
"""

import os
import sys
import numpy as np
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def draw_bbox_on_image(image, detections, save_path):
    """Draw bounding boxes on image and save."""
    from PIL import Image, ImageDraw
    
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
    draw = ImageDraw.Draw(img)
    
    # Colors for different classes
    colors = {
        'charger': (0, 255, 0),     # Green
        'plant': (255, 0, 0),        # Red
        'obstacle': (128, 128, 128), # Gray
        'unknown': (255, 255, 0),    # Yellow
    }
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox_2d']
        cls = det['class']
        conf = det['confidence']
        dist = det['estimated_distance']
        color_score = det.get('color_score', 1.0)
        
        color = colors.get(cls, (255, 255, 0))
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Create label
        label = f"{cls}: {conf:.2f} ({dist:.1f}m)"
        
        # Calculate label position - INSIDE bbox if near top edge
        label_y = y1 - 20
        if label_y < 5:
            label_y = y1 + 5  # Put inside bbox
        
        # Draw background for label
        try:
            bbox_label = draw.textbbox((x1, label_y), label)
            draw.rectangle(bbox_label, fill=(0, 0, 0))
        except:
            pass  # Skip if textbbox not available
        draw.text((x1, label_y), label, fill=color)
    
    img.save(save_path)
    return img


def main():
    print("="*60)
    print("RGB-Only Perception Test")
    print("Detects: charger (GREEN), plant (RED), obstacles (GRAY)")
    print("="*60)
    
    # Initialize Isaac Sim
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1280,
        "height": 720,
    })
    
    try:
        # Setup scene
        print("\n[1/5] Setting up scene...")
        from scene.setup_environment import setup_scene
        from scene.robot_interface import IsaacSimRobot
        from pxr import UsdGeom, Gf
        from omni.isaac.core.utils.types import ArticulationAction
        
        scene = setup_scene('partial', seed=42)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        
        # Create camera
        print("\n[2/5] Creating robot camera...")
        stage = world.stage
        camera_path = "/World/Turtlebot/base_link/robot_camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        
        camera_prim.CreateFocalLengthAttr(24.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.20))
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(90.0, 0.0, -90.0))
        
        # Create robot interface
        robot_interface = IsaacSimRobot(
            robot=robot, world=world, 
            camera_path=camera_path, objects=objects
        )
        
        # Initialize perception
        print("\n[3/5] Loading perception module...")
        from perception.vision_model import PerceptionModule
        
        perception = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompts={
                'charger': 'green box . green cube',
                'plant': 'red box . red cube',
                'obstacle': 'gray box . gray cube . white box',
            },
            temperature=2.0,
            threshold=0.3,
        )
        
        # Let physics settle
        print("\n[4/5] Settling physics...")
        for _ in range(50):
            world.step(render=True)
        
        # Test sequence - move around to find charger
        print("\n[5/5] Running perception test...")
        print("\nScene layout:")
        print("  Robot starts at (0, 0)")
        print("  Plant (red) at (2.0, 0.0) - blocking view")
        print("  Charger (green) at (2.8, 0.0) - behind plant")
        print("  Distractors (gray) at (2.0, ±1.5)")
        print("\nStrategy: Move sideways to see around the plant")
        
        all_detections = []
        test_num = 0
        
        def capture_and_detect(label):
            nonlocal test_num
            test_num += 1
            pose = robot_interface.get_pose()
            print(f"\n[{label}] Robot at ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
            
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            print(f"  Detections: {len(detections)}")
            for det in detections:
                print(f"    {det['class']}: conf={det['confidence']:.2f}, color={det['color_score']:.2f}, dist={det['estimated_distance']:.1f}m, pos=({det['world_position'][0]:.1f}, {det['world_position'][1]:.1f})")
                all_detections.append(det)
            
            save_path = os.path.join(project_root, f"test_{test_num}.png")
            draw_bbox_on_image(rgb, detections, save_path)
            print(f"  Saved: {save_path}")
            return detections
        
        def move_forward(steps=40):
            for _ in range(steps):
                robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([5.0, 5.0])))
                world.step(render=True)
            robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
            for _ in range(10):
                world.step(render=True)
        
        def turn(direction, steps=60):
            """direction: 'left' or 'right'"""
            if direction == 'left':
                vel = np.array([-3.0, 3.0])
            else:
                vel = np.array([3.0, -3.0])
            for _ in range(steps):
                robot.apply_wheel_actions(ArticulationAction(joint_velocities=vel))
                world.step(render=True)
            robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
            for _ in range(10):
                world.step(render=True)
        
        # Phase 1: Initial view (plant blocking charger)
        print("\n--- Phase 1: Initial View ---")
        capture_and_detect("Start")
        
        # Phase 2: Move left to see around plant
        print("\n--- Phase 2: Move Left ---")
        turn('left', 45)  # Turn left 45 degrees
        capture_and_detect("After turn left")
        move_forward(60)
        capture_and_detect("After move")
        
        # Phase 3: Turn back toward charger
        print("\n--- Phase 3: Turn toward charger ---")
        turn('right', 45)  # Turn right to face forward again
        capture_and_detect("Face forward")
        
        # Phase 4: Look for charger from this angle
        print("\n--- Phase 4: Look for charger ---")
        turn('right', 30)  # Turn slightly right
        capture_and_detect("Looking right")
        move_forward(40)
        capture_and_detect("Closer")
        
        # Phase 5: Check other side (where distractors are)
        print("\n--- Phase 5: Check right side (distractors) ---")
        turn('right', 60)
        capture_and_detect("Right side")
        
        # Summary
        print("\n" + "="*60)
        print("Detection Summary")
        print("="*60)
        
        charger_dets = [d for d in all_detections if d['class'] == 'charger']
        plant_dets = [d for d in all_detections if d['class'] == 'plant']
        obstacle_dets = [d for d in all_detections if d['class'] == 'obstacle']
        
        print(f"Total detections: {len(all_detections)}")
        print(f"  Charger (green): {len(charger_dets)}")
        print(f"  Plant (red): {len(plant_dets)}")
        print(f"  Obstacles (gray): {len(obstacle_dets)}")
        
        if charger_dets:
            print("\n🎉 CHARGER FOUND!")
            for det in charger_dets:
                print(f"  Est. position: ({det['world_position'][0]:.2f}, {det['world_position'][1]:.2f})")
                print(f"  Distance: {det['estimated_distance']:.2f}m")
        else:
            print("\n⚠ Charger not detected (may be occluded)")
        
        if plant_dets:
            print("\n🌱 PLANT DETECTIONS:")
            for i, det in enumerate(plant_dets[:3]):  # Show first 3
                print(f"  {i+1}. pos=({det['world_position'][0]:.2f}, {det['world_position'][1]:.2f}), dist={det['estimated_distance']:.1f}m")
        
        if obstacle_dets:
            print("\n📦 OBSTACLE DETECTIONS:")
            for i, det in enumerate(obstacle_dets[:3]):
                print(f"  {i+1}. pos=({det['world_position'][0]:.2f}, {det['world_position'][1]:.2f}), dist={det['estimated_distance']:.1f}m")
        
        print("\n📍 Ground Truth:")
        gt = robot_interface.get_object_ground_truth()
        for name, pos in gt.items():
            print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        print("\n" + "="*60)
        print("✅ Test Complete!")
        print("="*60)
        
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
