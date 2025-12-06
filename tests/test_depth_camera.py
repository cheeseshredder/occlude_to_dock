"""
Test Intel RealSense D455 - v7 SIMPLE (No Replicator)

Uses ONLY viewport capture (which works) and computes depth from stereo.
NO replicator, NO annotators - just simple viewport capture.
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("INTEL REALSENSE D455 - SIMPLE STEREO DEPTH")
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
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from pxr import UsdGeom, Gf, UsdLux, Usd, UsdPhysics
        from PIL import Image
        import cv2
        
        output_dir = os.path.join(project_root, "depth_camera_test")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n[1/6] Creating world...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = world.stage
        
        UsdLux.DistantLight.Define(stage, "/World/sunLight").CreateIntensityAttr(1000.0)
        UsdLux.DomeLight.Define(stage, "/World/domeLight").CreateIntensityAttr(300.0)
        
        print("\n[2/6] Spawning test objects...")
        
        charger_pos = np.array([2.8, 0.0, 0.25])
        world.scene.add(FixedCuboid(
            prim_path="/World/charger", name="charger",
            position=charger_pos, scale=np.array([0.3, 0.2, 0.5]),
            size=1.0, color=np.array([0.0, 0.8, 0.0]),
        ))
        
        plant_pos = np.array([2.0, 0.0, 0.5])
        world.scene.add(FixedCuboid(
            prim_path="/World/plant", name="plant",
            position=plant_pos, scale=np.array([0.4, 0.4, 1.0]),
            size=1.0, color=np.array([0.8, 0.0, 0.0]),
        ))
        
        obstacle_pos = np.array([2.0, 1.5, 0.225])
        world.scene.add(FixedCuboid(
            prim_path="/World/obstacle", name="obstacle",
            position=obstacle_pos, scale=np.array([0.35, 0.25, 0.45]),
            size=1.0, color=np.array([0.5, 0.5, 0.5]),
        ))
        
        print("  ✓ Objects spawned")
        
        print("\n[3/6] Spawning TurtleBot...")
        turtlebot_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd"
        add_reference_to_stage(usd_path=turtlebot_usd, prim_path="/World/Turtlebot")
        
        robot = world.scene.add(WheeledRobot(
            prim_path="/World/Turtlebot", name="turtlebot",
            wheel_dof_names=["wheel_left_joint", "wheel_right_joint"],
            create_robot=False, position=np.array([0.0, 0.0, 0.0]),
        ))
        print("  ✓ TurtleBot spawned")
        
        print("\n[4/6] Loading RealSense D455...")
        realsense_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Sensors/Intel/RealSense/rsd455.usd"
        add_reference_to_stage(usd_path=realsense_usd, prim_path="/World/RealSense")
        
        realsense_prim = stage.GetPrimAtPath("/World/RealSense")
        xform = UsdGeom.Xformable(realsense_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.05, 0.0, 0.2))
        xform.AddRotateYXZOp().Set(Gf.Vec3f(0, 0, 0))
        
        # Disable physics
        for prim in Usd.PrimRange(realsense_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
        
        print("  ✓ RealSense positioned")
        
        print("\n[5/6] Initializing...")
        world.reset()
        
        # Warm up
        for _ in range(60):
            world.step(render=True)
        
        print("\n[6/6] Capturing stereo images...")
        
        # Camera paths
        left_cam = "/World/RealSense/RSD455/Camera_OmniVision_OV9782_Left"
        right_cam = "/World/RealSense/RSD455/Camera_OmniVision_OV9782_Right"
        color_cam = "/World/RealSense/RSD455/Camera_OmniVision_OV9782_Color"
        
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        viewport = get_active_viewport()
        
        # Capture left
        print("  Capturing left...")
        viewport.set_active_camera(left_cam)
        for _ in range(20):
            world.step(render=True)
        left_path = os.path.join(output_dir, "stereo_left.png")
        capture_viewport_to_file(viewport, left_path)
        
        # Capture right
        print("  Capturing right...")
        viewport.set_active_camera(right_cam)
        for _ in range(20):
            world.step(render=True)
        right_path = os.path.join(output_dir, "stereo_right.png")
        capture_viewport_to_file(viewport, right_path)
        
        # Capture color
        print("  Capturing color...")
        viewport.set_active_camera(color_cam)
        for _ in range(20):
            world.step(render=True)
        color_path = os.path.join(output_dir, "stereo_color.png")
        capture_viewport_to_file(viewport, color_path)
        
        print("  ✓ Images captured")
        
        # Compute stereo depth
        print("\n" + "="*60)
        print("COMPUTING STEREO DEPTH")
        print("="*60)
        
        # Load images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is not None and right_img is not None:
            print(f"\n  Left:  {left_img.shape}")
            print(f"  Right: {right_img.shape}")
            
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Stereo matching using StereoBM
            print("\n  Computing disparity with StereoBM...")
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
            disparity = stereo.compute(left_gray, right_gray)
            
            # Normalize for visualization
            disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)
            
            # Save disparity
            cv2.imwrite(os.path.join(output_dir, "disparity_bm.png"), disp_vis)
            print(f"  ✓ Saved disparity_bm.png")
            
            # Try StereoSGBM for better results
            print("  Computing disparity with StereoSGBM...")
            stereo_sgbm = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=64,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
            disparity_sgbm = stereo_sgbm.compute(left_gray, right_gray)
            
            disp_sgbm_vis = cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX)
            disp_sgbm_vis = np.uint8(disp_sgbm_vis)
            
            cv2.imwrite(os.path.join(output_dir, "disparity_sgbm.png"), disp_sgbm_vis)
            print(f"  ✓ Saved disparity_sgbm.png")
            
            # Convert disparity to depth (approximate)
            # depth = baseline * focal_length / disparity
            # RealSense D455 baseline is ~95mm
            baseline = 0.095  # meters
            focal_length_px = 600  # approximate, depends on resolution
            
            # Avoid division by zero
            disparity_float = disparity_sgbm.astype(np.float32) / 16.0  # SGBM returns fixed point
            disparity_float[disparity_float <= 0] = 0.1
            
            depth = baseline * focal_length_px / disparity_float
            depth[depth > 10] = 10  # Clip to 10m max
            
            # Save depth
            np.save(os.path.join(output_dir, "depth_stereo.npy"), depth)
            
            depth_vis = (depth / 10.0 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "depth_stereo.png"), depth_vis)
            print(f"  ✓ Saved depth_stereo.png and depth_stereo.npy")
            
            # Analysis
            print("\n" + "="*60)
            print("DEPTH ANALYSIS")
            print("="*60)
            
            h, w = depth.shape
            center_depth = depth[h//2, w//2]
            
            cam_pos = np.array([0.05, 0.0, 0.2])
            expected_plant = np.linalg.norm(plant_pos - cam_pos)
            expected_charger = np.linalg.norm(charger_pos - cam_pos)
            
            print(f"\n  Center depth: {center_depth:.2f}m")
            print(f"  Expected (plant): {expected_plant:.2f}m")
            print(f"  Expected (charger): {expected_charger:.2f}m")
            
            # Color-coded depth map
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(output_dir, "depth_colormap.png"), depth_color)
            print(f"  ✓ Saved depth_colormap.png")
            
        else:
            print("  ✗ Could not load stereo images")
        
        # List output files
        print("\n" + "="*60)
        print("OUTPUT FILES")
        print("="*60)
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size/1024:.1f} KB)")
        
        print("\n" + "="*60)
        print("🎮 Running... Ctrl+C to exit")
        print("="*60)
        
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
