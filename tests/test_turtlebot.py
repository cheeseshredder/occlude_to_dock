"""
Test TurtleBot3 with Robot-View Camera (FIXED - centered, looking forward)
"""

import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    print("="*60)
    print("TurtleBot3 + Robot Camera View Test")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        print("\n[1] Setting up scene...")
        from scene.setup_environment import setup_scene
        from omni.isaac.core.utils.types import ArticulationAction
        
        scene = setup_scene('partial', seed=42)
        
        world = scene['world']
        robot = scene['robot']
        
        print(f"\n[2] Creating camera on robot...")
        
        from pxr import UsdGeom, Gf
        stage = world.stage
        
        # Create camera under robot base_link
        camera_path = "/World/Turtlebot/base_link/robot_camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        
        # Camera properties
        camera_prim.CreateFocalLengthAttr(24.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateVerticalApertureAttr(15.2908)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        
        # CENTERED on robot, 20cm up from base
        # X=0 (centered), Y=0 (centered), Z=0.2 (20cm up)
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.20))
        
        # Camera default looks down -Z axis
        # Robot forward is +X axis
        # We need to rotate camera to look along +X
        # 
        # From your working values: Rotate X=90, Y=0, Z=-90
        # Let's try different combinations
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(90.0, 0.0, -90.0))  # Look forward (+X)
        
        print(f"  ✓ Camera at {camera_path}")
        print(f"    Translate: [0, 0, 0.2]")
        print(f"    Rotate XYZ: [90, 0, -90]")
        
        # Settle physics
        print("\n[3] Settling physics...")
        for _ in range(50):
            world.step(render=True)
        
        # Check robot position
        pos, rot = robot.get_world_pose()
        print(f"  Robot position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        # Switch viewport to robot camera
        print("\n[4] Switching viewport to robot camera...")
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        
        viewport = get_active_viewport()
        viewport.set_active_camera(camera_path)
        
        # Render frames
        for _ in range(30):
            world.step(render=True)
        
        # Capture
        print("\n[5] Capturing robot view (at start)...")
        output_path = os.path.join(project_root, "robot_view_1.png")
        capture_viewport_to_file(viewport, output_path)
        print(f"  ✓ Saved: {output_path}")
        
        # Drive toward charger
        print("\n[6] Driving toward charger...")
        for i in range(300):  # More steps to get closer
            robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([5.0, 5.0])))
            world.step(render=True)
        
        pos_end, _ = robot.get_world_pose()
        print(f"  New position: [{pos_end[0]:.2f}, {pos_end[1]:.2f}]")
        
        # Capture after move
        print("\n[7] Capturing robot view (after move)...")
        for _ in range(20):
            world.step(render=True)
        
        output_path2 = os.path.join(project_root, "robot_view_2.png")
        capture_viewport_to_file(viewport, output_path2)
        print(f"  ✓ Saved: {output_path2}")
        
        # Stop
        robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
        
        print("\n" + "="*60)
        print("✅ Test Complete!")
        print("="*60)
        print("\nIf camera still wrong, manually adjust rotation in GUI")
        print("and note down the working values!")
        
        print("\nKeep window open - Ctrl+C to close...")
        while simulation_app.is_running():
            world.step(render=True)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
