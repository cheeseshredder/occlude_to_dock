"""
Isaac Sim Launcher for Occlude-to-Dock POMDP System
Run with: isaac-python.bat run_isaac_sim.py
"""

import os
import sys
import numpy as np
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("Occlude-to-Dock: Isaac Sim POMDP Demo")
    print("="*60)
    
    # Initialize Isaac Sim
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({
        "headless": False,
        "width": 1280,
        "height": 720,
        "window_width": 1920,
        "window_height": 1080,
    })
    
    try:
        # Setup scene
        print("\n[1/5] Setting up scene...")
        from scene.setup_environment import setup_scene
        scene = setup_scene('partial', seed=42)
        
        world = scene['world']
        robot = scene['robot']
        camera_path = scene['camera_path']
        objects = scene['objects']
        
        # Create robot interface
        print("\n[2/5] Creating robot interface...")
        from scene.robot_interface import IsaacSimRobot
        robot_interface = IsaacSimRobot(
            robot=robot,
            world=world,
            camera_path=camera_path,
            objects=objects
        )
        
        # Switch viewport to robot camera
        print("\n[3/5] Switching to robot camera view...")
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        viewport.set_active_camera(camera_path)
        
        # Let physics settle
        print("\n[4/5] Settling physics...")
        for _ in range(50):
            world.step(render=True)
        
        # Test basic operations
        print("\n[5/5] Running test sequence...")
        
        # Get initial pose
        pose = robot_interface.get_pose()
        print(f"  Initial pose: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={np.degrees(pose[2]):.1f}°")
        
        # Get camera data
        print("  Capturing RGB + Depth...")
        rgb, depth = robot_interface.get_camera_data()
        print(f"  RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
        
        # Save test images
        from PIL import Image
        
        rgb_path = os.path.join(project_root, "test_rgb.png")
        Image.fromarray(rgb).save(rgb_path)
        print(f"  Saved: {rgb_path}")
        
        # Save depth as colormap
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_colored = (plt_colormap(depth_normalized) * 255).astype(np.uint8)
        depth_path = os.path.join(project_root, "test_depth.png")
        Image.fromarray(depth_colored).save(depth_path)
        print(f"  Saved: {depth_path}")
        
        print("\n" + "="*60)
        print("✅ Isaac Sim Setup Complete!")
        print("="*60)
        print("\nControls:")
        print("  - The robot is ready for POMDP navigation")
        print("  - Press Ctrl+C to exit")
        print("\nNext steps:")
        print("  - Run main_isaac.py for full POMDP demo")
        
        # Keep running
        print("\nSimulation running...")
        while simulation_app.is_running():
            world.step(render=True)
            
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()


def plt_colormap(data):
    """Simple viridis-like colormap without matplotlib."""
    # Viridis-inspired colors
    colors = np.array([
        [0.267, 0.004, 0.329],  # Dark purple
        [0.282, 0.140, 0.457],  # Purple
        [0.254, 0.265, 0.529],  # Blue-purple
        [0.206, 0.371, 0.553],  # Blue
        [0.163, 0.471, 0.558],  # Teal
        [0.127, 0.566, 0.550],  # Cyan
        [0.134, 0.658, 0.517],  # Green-cyan
        [0.267, 0.749, 0.440],  # Green
        [0.477, 0.821, 0.318],  # Yellow-green
        [0.741, 0.873, 0.150],  # Yellow
        [0.993, 0.906, 0.144],  # Bright yellow
    ])
    
    # Interpolate
    indices = data * (len(colors) - 1)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    t = indices - lower
    
    lower = np.clip(lower, 0, len(colors) - 1)
    upper = np.clip(upper, 0, len(colors) - 1)
    
    result = colors[lower] * (1 - t[..., None]) + colors[upper] * t[..., None]
    return result


if __name__ == "__main__":
    main()
