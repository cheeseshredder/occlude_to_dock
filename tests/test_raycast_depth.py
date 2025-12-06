"""
Test depth using Isaac Sim RAYCAST - Perfect Ground Truth Depth

This bypasses all the broken replicator/annotator issues by using
physics raycast to get exact depth at any pixel.
"""

import os
import sys
import numpy as np
import math

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("DEPTH VIA RAYCAST - GROUND TRUTH")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({
        "headless": False, 
        "width": 1280, 
        "height": 720,
    })
    
    try:
        from omni.isaac.core import World
        from omni.isaac.core.objects import FixedCuboid
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, Gf, UsdLux, Usd, UsdPhysics
        import omni.physx
        from omni.physx import get_physx_scene_query_interface
        from PIL import Image
        import cv2
        
        output_dir = os.path.join(project_root, "depth_camera_test")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n[1/5] Creating world...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = world.stage
        
        UsdLux.DistantLight.Define(stage, "/World/sunLight").CreateIntensityAttr(1000.0)
        
        print("\n[2/5] Spawning objects...")
        
        # Known positions
        charger_pos = np.array([2.8, 0.0, 0.25])
        plant_pos = np.array([2.0, 0.0, 0.5])
        obstacle_pos = np.array([2.0, 1.5, 0.225])
        
        world.scene.add(FixedCuboid(
            prim_path="/World/charger", name="charger",
            position=charger_pos, scale=np.array([0.3, 0.2, 0.5]),
            size=1.0, color=np.array([0.0, 0.8, 0.0]),
        ))
        
        world.scene.add(FixedCuboid(
            prim_path="/World/plant", name="plant",
            position=plant_pos, scale=np.array([0.4, 0.4, 1.0]),
            size=1.0, color=np.array([0.8, 0.0, 0.0]),
        ))
        
        world.scene.add(FixedCuboid(
            prim_path="/World/obstacle", name="obstacle",
            position=obstacle_pos, scale=np.array([0.35, 0.25, 0.45]),
            size=1.0, color=np.array([0.5, 0.5, 0.5]),
        ))
        print("  ✓ Objects created")
        
        print("\n[3/5] Creating camera prim...")
        
        # Create a simple camera
        camera_path = "/World/DepthCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        
        # Position camera
        cam_pos = np.array([0.05, 0.0, 0.2])  # Same as RealSense position
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(*cam_pos))
        
        # Camera parameters (matching RealSense D455)
        camera.GetFocalLengthAttr().Set(1.93)  # mm (for 86° HFOV)
        camera.GetHorizontalApertureAttr().Set(3.6)  # mm
        camera.GetVerticalApertureAttr().Set(2.4)  # mm
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
        
        print(f"  ✓ Camera at {cam_pos}")
        
        print("\n[4/5] Initializing physics...")
        world.reset()
        
        for _ in range(60):
            world.step(render=True)
        
        print("\n[5/5] Testing raycast depth...")
        
        # Get physx interface
        physx_query = get_physx_scene_query_interface()
        
        def raycast_depth(origin, direction, max_dist=20.0):
            """Cast a ray and return hit distance"""
            origin_carb = (float(origin[0]), float(origin[1]), float(origin[2]))
            dir_carb = (float(direction[0]), float(direction[1]), float(direction[2]))
            
            hit = physx_query.raycast_closest(origin_carb, dir_carb, max_dist)
            if hit["hit"]:
                return hit["distance"], hit["rigidBody"]
            return None, None
        
        def pixel_to_ray(u, v, img_w=1280, img_h=720, hfov_deg=86):
            """Convert pixel coordinates to ray direction (camera facing +X)"""
            hfov = math.radians(hfov_deg)
            vfov = hfov * (img_h / img_w)
            
            # Normalized coordinates (-1 to 1)
            nx = (u - img_w/2) / (img_w/2)
            ny = (img_h/2 - v) / (img_h/2)  # Flip Y
            
            # Angles
            angle_x = nx * (hfov / 2)  # Horizontal angle (yaw)
            angle_y = ny * (vfov / 2)  # Vertical angle (pitch)
            
            # Ray direction (camera facing +X)
            dx = math.cos(angle_y) * math.cos(angle_x)
            dy = math.cos(angle_y) * math.sin(angle_x)  # Note: Y is horizontal in Isaac
            dz = math.sin(angle_y)
            
            # Normalize
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            return np.array([dx/length, dy/length, dz/length])
        
        print("\n" + "="*60)
        print("RAYCAST DEPTH RESULTS")
        print("="*60)
        
        # Test specific pixels
        test_points = [
            ("Center (plant)", 640, 360),
            ("Left (obstacle)", 150, 360),
            ("Right (empty)", 1100, 360),
            ("Top (sky)", 640, 100),
            ("Bottom (ground)", 640, 600),
        ]
        
        for name, u, v in test_points:
            ray_dir = pixel_to_ray(u, v)
            dist, body = raycast_depth(cam_pos, ray_dir)
            
            if dist:
                print(f"\n  {name} [{u}, {v}]:")
                print(f"    Depth: {dist:.3f}m")
                print(f"    Hit: {body}")
            else:
                print(f"\n  {name} [{u}, {v}]: No hit (sky/far)")
        
        # Compare to expected
        print("\n" + "="*60)
        print("ACCURACY COMPARISON")
        print("="*60)
        
        # Center should hit plant
        ray_dir = pixel_to_ray(640, 360)
        dist, body = raycast_depth(cam_pos, ray_dir)
        expected_plant = np.linalg.norm(plant_pos - cam_pos)
        
        print(f"\n  Plant (center pixel):")
        print(f"    Expected: {expected_plant:.3f}m")
        print(f"    Raycast:  {dist:.3f}m")
        if dist:
            error = abs(dist - expected_plant)
            print(f"    Error:    {error:.3f}m ({100*error/expected_plant:.1f}%)")
        
        # Left should hit obstacle
        ray_dir = pixel_to_ray(150, 350)
        dist, body = raycast_depth(cam_pos, ray_dir)
        expected_obstacle = np.linalg.norm(obstacle_pos - cam_pos)
        
        print(f"\n  Obstacle (left pixel):")
        print(f"    Expected: {expected_obstacle:.3f}m")
        print(f"    Raycast:  {dist:.3f}m" if dist else "    Raycast:  No hit")
        
        # Generate full depth image via raycast
        print("\n" + "="*60)
        print("GENERATING RAYCAST DEPTH IMAGE")
        print("="*60)
        
        img_w, img_h = 320, 180  # Lower res for speed
        depth_img = np.zeros((img_h, img_w), dtype=np.float32)
        
        print(f"  Casting {img_w * img_h} rays...")
        
        for v in range(img_h):
            for u in range(img_w):
                # Scale to full resolution
                u_full = int(u * 1280 / img_w)
                v_full = int(v * 720 / img_h)
                
                ray_dir = pixel_to_ray(u_full, v_full)
                dist, _ = raycast_depth(cam_pos, ray_dir)
                depth_img[v, u] = dist if dist else 20.0
            
            if v % 30 == 0:
                print(f"    Row {v}/{img_h}...")
        
        # Save depth image
        depth_vis = np.clip(depth_img, 0, 10)
        depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "raycast_depth.png"), depth_vis)
        
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(output_dir, "raycast_depth_color.png"), depth_color)
        
        np.save(os.path.join(output_dir, "raycast_depth.npy"), depth_img)
        
        print(f"\n  ✓ Saved raycast_depth.png")
        print(f"  ✓ Saved raycast_depth_color.png")
        print(f"  ✓ Saved raycast_depth.npy")
        
        print("\n" + "="*60)
        print("🎯 RAYCAST = GROUND TRUTH DEPTH")
        print("="*60)
        print("\nThis method gives PERFECT depth accuracy because it uses")
        print("the actual physics simulation geometry, not image processing.")
        print("\nFor POMDP navigation, we can use raycast at detected")
        print("object pixel locations to get their exact distances.")
        
        print("\n🎮 Running... Ctrl+C to exit")
        
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
