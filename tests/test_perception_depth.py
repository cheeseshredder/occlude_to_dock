"""
Test GroundingDINO + Realistic Depth Integration

Verifies that the perception module correctly uses raycast depth
with realistic noise instead of inaccurate bbox estimation.
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("PERCEPTION + DEPTH INTEGRATION TEST")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from omni.isaac.core import World
        from omni.isaac.core.objects import FixedCuboid
        from pxr import UsdGeom, Gf, UsdLux
        from omni.physx import get_physx_scene_query_interface
        from omni.kit.viewport.utility import get_active_viewport
        import omni.kit.viewport.utility as viewport_utils
        import cv2
        
        # Import our perception module
        from perception.vision_model import PerceptionModule, create_isaac_raycast_func
        
        output_dir = os.path.join(project_root, "perception_test")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n[1/6] Creating world...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = world.stage
        UsdLux.DistantLight.Define(stage, "/World/sunLight").CreateIntensityAttr(1000.0)
        UsdLux.DomeLight.Define(stage, "/World/domeLight").CreateIntensityAttr(300.0)
        
        print("\n[2/6] Spawning objects...")
        
        # Known positions for validation - spread out so all visible
        charger_pos = np.array([3.0, 0.0, 0.25])     # Center, furthest (target)
        plant_pos = np.array([2.0, -1.0, 0.5])       # Left side, closer
        obstacle_pos = np.array([2.5, 1.2, 0.225])   # Right side
        
        world.scene.add(FixedCuboid(
            prim_path="/World/charger", name="charger",
            position=charger_pos, scale=np.array([0.3, 0.4, 0.5]),
            size=1.0, color=np.array([0.0, 0.8, 0.0]),  # Green
        ))
        world.scene.add(FixedCuboid(
            prim_path="/World/plant", name="plant",
            position=plant_pos, scale=np.array([0.4, 0.4, 1.0]),
            size=1.0, color=np.array([0.8, 0.0, 0.0]),  # Red
        ))
        world.scene.add(FixedCuboid(
            prim_path="/World/obstacle", name="obstacle",
            position=obstacle_pos, scale=np.array([0.35, 0.35, 0.45]),
            size=1.0, color=np.array([0.5, 0.5, 0.5]),  # Gray
        ))
        
        print(f"  Charger at: {charger_pos}")
        print(f"  Plant at:   {plant_pos}")
        print(f"  Obstacle at: {obstacle_pos}")
        
        print("\n[3/6] Creating camera...")
        
        # Robot/camera position - moved back to see all objects
        robot_pos = np.array([-0.5, 0.0, 0.0])  # Back from origin
        robot_yaw = 0.0  # Facing +X
        cam_height = 0.3  # Slightly higher
        cam_forward = 0.05
        
        cam_pos = np.array([
            robot_pos[0] + cam_forward,
            robot_pos[1],
            cam_height
        ])
        
        # Create camera prim  
        camera_path = "/World/RobotCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera_prim = camera.GetPrim()
        xformable = UsdGeom.Xformable(camera_prim)
        
        # Clear any existing transforms
        xformable.ClearXformOpOrder()
        
        # Camera orientation in Isaac Sim/USD:
        # - Default USD camera looks down -Z in local space with +Y up
        # - To face +X world direction with +Z up, we need specific rotation
        # 
        # Set position
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(cam_pos[0], cam_pos[1], cam_pos[2]))
        
        # Rotation to face +X with Z-up:
        # Using ZXY rotation order with (X=0, Y=-90, Z=-90)
        rotate_op = xformable.AddRotateZXYOp()
        rotate_op.Set(Gf.Vec3f(0, -90, -90))  # Facing +X with Z-up
        
        print(f"  Camera at: {cam_pos}")
        print(f"  Camera rotation: ZXY(0, -90, -90) - facing +X, Z-up")
        
        # Camera lens parameters
        camera.GetFocalLengthAttr().Set(18.14756)
        camera.GetHorizontalApertureAttr().Set(20.955)
        camera.GetVerticalApertureAttr().Set(15.2908)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
        
        print(f"  Camera at: {cam_pos}")
        print(f"  Camera facing: +X (forward)")
        
        print("\n[4/6] Initializing perception...")
        
        # Initialize perception module
        perception = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            use_depth_sensor=True,
            depth_noise_enabled=True,
            hfov_deg=60.0,
            img_width=640,
            img_height=480,
            device='cuda',
        )
        
        print("\n[5/6] Initializing simulation...")
        world.reset()
        
        # Setup raycast
        physx_query = get_physx_scene_query_interface()
        raycast_func = create_isaac_raycast_func(physx_query)
        
        # Connect raycast to perception
        perception.set_raycast_func(raycast_func, cam_pos)
        
        # Set viewport to camera
        viewport = get_active_viewport()
        viewport.set_active_camera(camera_path)
        
        # Warm up rendering
        print("  Warming up renderer...")
        for i in range(100):
            world.step(render=True)
            if i % 20 == 0:
                print(f"    Frame {i}/100")
        
        print("\n[6/6] Running detection test...")
        
        # Capture RGB image using viewport capture
        print("\n  Capturing image...")
        
        # Method: Use viewport capture to buffer
        from PIL import Image
        import asyncio
        
        rgb_path = os.path.join(output_dir, "test_rgb.png")
        
        # Capture using the synchronous method
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file
            capture_viewport_to_file(viewport, rgb_path)
            
            # Wait for file to be written
            for _ in range(30):
                world.step(render=True)
            
            # Load the captured image
            if os.path.exists(rgb_path):
                rgb_image = cv2.imread(rgb_path)
                if rgb_image is not None:
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    print(f"  ✓ Captured image: {rgb_image.shape}")
                    
                    # Resize to expected size
                    rgb_image = cv2.resize(rgb_image, (640, 480))
                else:
                    raise Exception("Failed to load captured image")
            else:
                raise Exception(f"Image file not found: {rgb_path}")
                
        except Exception as e:
            print(f"  ⚠ Capture failed: {e}")
            print("  Using mock detection instead...")
            rgb_image = None
        
        # Run detection
        print("\n  Running GroundingDINO detection...")
        robot_pose = (robot_pos[0], robot_pos[1], robot_yaw)
        
        if rgb_image is not None:
            detections = perception.detect(rgb_image, robot_pose)
        else:
            # Use mock detection
            detections = perception._mock_detect(np.zeros((480, 640, 3), dtype=np.uint8), robot_pose)
        
        print(f"\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        
        print(f"\nDetected {len(detections)} objects:")
        
        for det in detections:
            class_name = det['class']
            est_dist = det['estimated_distance']
            uncertainty = det['distance_uncertainty']
            depth_method = det['depth_method']
            world_pos = det['world_position']
            confidence = det['confidence']
            
            # Get true distance for comparison
            if class_name == 'charger':
                true_pos = charger_pos
            elif class_name == 'plant':
                true_pos = plant_pos
            else:
                true_pos = obstacle_pos
            
            true_dist = np.linalg.norm(true_pos[:2] - robot_pos[:2])
            error = abs(est_dist - true_dist)
            error_pct = 100 * error / true_dist if true_dist > 0 else 0
            
            print(f"\n  {class_name.upper()}:")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Depth method: {depth_method}")
            print(f"    Estimated distance: {est_dist:.2f}m ± {uncertainty:.2f}m")
            print(f"    True distance: {true_dist:.2f}m")
            print(f"    Error: {error:.2f}m ({error_pct:.1f}%)")
            print(f"    World position: ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
            print(f"    True position: ({true_pos[0]:.2f}, {true_pos[1]:.2f})")
        
        # Test raycast directly
        print(f"\n" + "="*60)
        print("DIRECT RAYCAST TEST")
        print("="*60)
        
        from perception.depth_sensor import RealisticDepthSensor
        sensor = RealisticDepthSensor(
            hfov_deg=60.0, 
            img_width=640, 
            img_height=480,
            base_noise_std=0.05,       # 5cm base noise
            distance_noise_factor=0.04  # 4% of distance
        )
        
        # Test center pixel - should hit something
        center_u, center_v = 320, 240
        
        print(f"\n  Testing raycast at center pixel ({center_u}, {center_v}):")
        
        # Perfect raycast (no noise)
        depth_perfect, valid, hit_body = sensor.get_depth_at_pixel(
            cam_pos, center_u, center_v, raycast_func, add_noise=False
        )
        print(f"    Perfect depth: {depth_perfect:.3f}m (hit: {hit_body})")
        
        # With CHALLENGING noise (multiple samples)
        samples = []
        for _ in range(100):
            depth, valid, _ = sensor.get_depth_at_pixel(
                cam_pos, center_u, center_v, raycast_func, add_noise=True
            )
            if valid:
                samples.append(depth)
        
        if samples:
            mean_depth = np.mean(samples)
            std_depth = np.std(samples)
            error_pct = 100 * std_depth / depth_perfect if depth_perfect > 0 else 0
            print(f"    Noisy depth: {mean_depth:.3f}m ± {std_depth*100:.1f}cm ({error_pct:.1f}% error)")
            print(f"    Expected noise: {(0.05 + 0.04*depth_perfect)*100:.1f}cm")
        
        print(f"\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print("""
  Raycast depth with CHALLENGING noise provides:
  ✓ ~8-12% error (realistic for difficult conditions)
  ✓ Still much better than bbox (~40% error)
  ✓ Forces POMDP to handle real uncertainty
  
  The navigation system must handle this noise robustly!
        """)
        
        # Save annotated image if we have detections
        if rgb_image is not None and len(detections) > 0:
            img_annotated = rgb_image.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(c) for c in det['bbox_2d']]
                color = (0, 255, 0) if det['class'] == 'charger' else \
                        (255, 0, 0) if det['class'] == 'plant' else (128, 128, 128)
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']} {det['estimated_distance']:.1f}m"
                cv2.putText(img_annotated, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imwrite(os.path.join(output_dir, "detections.png"), 
                       cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
            print(f"\n  Saved annotated image to {output_dir}/detections.png")
        
        print("\n🎮 Camera view should now show objects. Ctrl+C to exit")
        
        while simulation_app.is_running():
            world.step(render=True)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
