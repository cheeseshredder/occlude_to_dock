"""
Belief Update Test with TELEPORTATION to known viewpoints.
Uses position averaging to handle noisy depth estimates.
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("BELIEF UPDATE WITH TELEPORTATION")
    print("Teleporting to known viewpoints where objects are visible")
    print("="*60)
    
    output_dir = os.path.join(project_root, "belief_maps_teleport")
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
        
        print("\n[1/4] Setting up scene...")
        scene = setup_scene('partial', seed=42)
        world = scene['world']
        robot = scene['robot']
        objects = scene['objects']
        
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
        
        print("\n[4/4] Creating semantic grid with averaging...")
        from mapping.semantic_grid import SemanticGrid
        
        grid = SemanticGrid(
            size_meters=(6.0, 6.0),
            resolution=0.5,
            origin=(-3.0, -3.0),
        )
        
        world.reset()
        for _ in range(30):
            world.step(render=True)
        
        def teleport_robot(x, y, yaw_deg):
            """Teleport robot to specific pose."""
            r = Rotation.from_euler('z', yaw_deg, degrees=True)
            quat = r.as_quat()
            robot.set_world_pose(
                position=np.array([x, y, 0.0]),
                orientation=np.array([quat[3], quat[0], quat[1], quat[2]])
            )
            for _ in range(20):
                world.step(render=True)
        
        def sense_and_update(label, step):
            """Capture and update belief."""
            pose = robot_interface.get_pose()
            rgb = robot_interface.get_rgb()
            detections = perception.detect(rgb, pose)
            
            grid.update_from_detections(detections, pose)
            
            print(f"\n--- {label} ---")
            print(f"Robot: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")
            print(f"Detections: {len(detections)}")
            
            for det in detections:
                marker = "🎯" if det['class'] == 'charger' else "  "
                print(f"  {marker} {det['class']}: conf={det['confidence']:.2f}, color={det.get('color_score', 0):.2f}, pos=({det['world_position'][0]:.1f}, {det['world_position'][1]:.1f})")
            
            # Save visualization
            vis = grid.visualize(pose)
            img = Image.fromarray(vis)
            img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
            
            draw = ImageDraw.Draw(img)
            draw.rectangle([5, 5, 130, 105], fill=(50, 50, 50))
            draw.rectangle([10, 10, 25, 25], fill=(0, 200, 0))
            draw.text((30, 10), "Charger", fill=(255, 255, 255))
            draw.rectangle([10, 30, 25, 45], fill=(200, 0, 0))
            draw.text((30, 30), "Plant", fill=(255, 255, 255))
            draw.rectangle([10, 50, 25, 65], fill=(100, 100, 100))
            draw.text((30, 50), "Obstacle", fill=(255, 255, 255))
            draw.rectangle([10, 70, 25, 85], fill=(0, 0, 255))
            draw.text((30, 70), "Robot", fill=(255, 255, 255))
            draw.text((10, img.height - 25), f"Step {step}: {label}", fill=(0, 0, 0))
            
            save_path = os.path.join(output_dir, f"belief_{step:02d}.png")
            img.save(save_path)
            
            # Also save raw camera image
            cam_img = Image.fromarray(rgb)
            cam_draw = ImageDraw.Draw(cam_img)
            colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (128, 128, 128)}
            for det in detections:
                x1, y1, x2, y2 = det['bbox_2d']
                color = colors.get(det['class'], (255, 255, 0))
                cam_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                cam_draw.text((x1, max(5, y1-15)), f"{det['class']}", fill=color)
            cam_path = os.path.join(output_dir, f"camera_{step:02d}.png")
            cam_img.save(cam_path)
            
            return detections
        
        # ============================================================
        # VIEWPOINT SEQUENCE
        # Based on manual testing, we know these viewpoints work
        # ============================================================
        
        print("\n" + "="*60)
        print("VIEWPOINT SEQUENCE")
        print("="*60)
        
        # Viewpoints: (x, y, yaw_degrees, label)
        viewpoints = [
            (0.0, 0.0, 0, "Start - facing plant"),
            (0.0, 1.0, -30, "Left side - should see charger"),
            (0.0, 1.2, -45, "More left - charger visible"),
            (0.3, 1.0, -40, "Adjusted"),
            (0.0, -1.0, 30, "Right side - distractor"),
            (0.0, -1.2, 45, "More right"),
            (1.0, 1.5, -60, "Far left"),
            (1.0, -1.5, 60, "Far right"),
            (0.5, 0.0, 0, "Center approach"),
            (1.0, 0.5, -20, "Angled approach"),
        ]
        
        step = 0
        all_detections = []
        
        for x, y, yaw, label in viewpoints:
            step += 1
            print(f"\n>>> Teleporting to ({x}, {y}), yaw={yaw}°")
            teleport_robot(x, y, yaw)
            
            dets = sense_and_update(label, step)
            all_detections.extend(dets)
        
        # ============================================================
        # RESULTS
        # ============================================================
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        grid.print_summary()
        
        # Get estimates
        charger_pos, charger_conf = grid.get_charger_estimate()
        plant_pos, plant_conf = grid.get_class_estimate('plant')
        obstacle_pos, obstacle_conf = grid.get_class_estimate('obstacle')
        
        print("\n📍 Ground Truth:")
        gt = robot_interface.get_object_ground_truth()
        for name, pos in gt.items():
            print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        print("\n📏 Position Errors:")
        if charger_pos:
            error = np.sqrt((charger_pos[0] - 2.8)**2 + (charger_pos[1] - 0.0)**2)
            print(f"  Charger: estimate=({charger_pos[0]:.2f}, {charger_pos[1]:.2f}), error={error:.2f}m")
        else:
            print("  Charger: NOT DETECTED")
            
        if plant_pos:
            error = np.sqrt((plant_pos[0] - 2.0)**2 + (plant_pos[1] - 0.0)**2)
            print(f"  Plant: estimate=({plant_pos[0]:.2f}, {plant_pos[1]:.2f}), error={error:.2f}m")
        
        # Count detections by class
        det_counts = {}
        for det in all_detections:
            cls = det['class']
            det_counts[cls] = det_counts.get(cls, 0) + 1
        
        print(f"\n📊 Detection counts: {det_counts}")
        print(f"💾 Saved to: {output_dir}/")
        
        print("\n" + "="*60)
        print("✅ Complete!")
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
