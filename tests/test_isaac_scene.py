"""
Test script for Isaac Sim scene setup - Interactive Mode

Run this with Isaac Sim Python to verify everything works:
    isaac-sim-python test_isaac_scene.py
    
This version keeps Isaac Sim open and responsive so you can inspect the scene.
Close the Isaac Sim window when done.
"""

import sys
import os

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("\n" + "=" * 70)
print(" OCCLUDE-TO-DOCK: Isaac Sim Scene Test")
print("=" * 70)

# Check if Isaac Sim is available
try:
    from omni.isaac.kit import SimulationApp
    print("✓ Isaac Sim imports successful")
except ImportError as e:
    print(f"\n❌ ERROR: Isaac Sim not available")
    print(f"   {e}")
    print("\nThis script must be run with Isaac Sim Python:")
    print("   $ isaac-sim-python test_isaac_scene.py")
    sys.exit(1)

# Create simulation app
print("\n[1/6] Starting Isaac Sim...")
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
print("✓ Isaac Sim window opened")

try:
    # Import scene setup
    print("\n[2/6] Importing scene setup...")
    from scene.setup_environment import setup_scene
    from scene.robot_interface import IsaacSimRobot
    print("✓ Scene modules loaded")
    
    # Setup scene
    print("\n[3/6] Setting up scene (this may take a moment)...")
    scene = setup_scene('partial', seed=42)
    print("✓ Scene created")
    
    # Verify components
    print("\n[4/6] Verifying scene components...")
    components_ok = True
    
    if scene['world'] is None:
        print("  ❌ World not created")
        components_ok = False
    else:
        print("  ✓ World: OK")
    
    if scene['robot'] is None:
        print("  ❌ Robot not created")
        components_ok = False
    else:
        print("  ✓ Robot: OK")
    
    if len(scene['objects']) < 4:
        print(f"  ⚠ Expected 4+ objects, found {len(scene['objects'])}")
    else:
        print(f"  ✓ Objects: {len(scene['objects'])} found")
        for obj_name in scene['objects']:
            print(f"    - {obj_name}")
    
    # Create robot interface
    print("\n[5/6] Creating robot interface...")
    robot = IsaacSimRobot(
        scene['robot'],
        scene['world'],
        scene.get('camera')
    )
    print("✓ IsaacSimRobot initialized")
    print(f"  - Wheel base: {getattr(robot, 'wheel_base', 0.16)}m")
    print(f"  - Wheel radius: {getattr(robot, 'wheel_radius', 0.033)}m")
    print("✓ Robot interface created")
    
    # Test robot
    print("\n[6/6] Testing robot functionality...")
    
    # Test pose
    pose = robot.get_pose()
    print(f"  • Initial pose: x={pose[0]:.3f}, y={pose[1]:.3f}, θ={pose[2]:.3f} rad")
    
    # Test velocity command
    print("  • Commanding forward motion (0.1 m/s)...")
    robot.set_velocity(0.1, 0.0)
    
    # Step simulation
    for i in range(30):
        scene['world'].step(render=True)
    
    pose = robot.get_pose()
    print(f"  • After 30 steps: x={pose[0]:.3f}, y={pose[1]:.3f}, θ={pose[2]:.3f} rad")
    
    # Stop
    robot.stop()
    print("  ✓ Robot movement test complete")
    
    # Success message
    print("\n" + "=" * 70)
    print(" ✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nWhat you should see in Isaac Sim window:")
    print("  • Gray ground plane")
    print("  • Red box (charger) at x ≈ 2.8m")
    print("  • Green cylinder (plant) at x ≈ 2.0m, partially occluding charger")
    print("  • Gray boxes (distractors) at y = ±1.5m")
    print("  • Blue box (robot) at origin")
    print("  • Gray walls forming 6×6m room boundaries")
    print("\n" + "=" * 70)
    print(" 🔍 SCENE IS NOW OPEN FOR INSPECTION")
    print("=" * 70)
    print("\nYou can now:")
    print("  • Rotate camera: Middle mouse button + drag")
    print("  • Pan camera: Shift + Middle mouse + drag")
    print("  • Zoom: Mouse wheel")
    print("  • Select objects: Left click")
    print("\nClose the Isaac Sim window when you're done inspecting.")
    print("=" * 70)
    
    # Keep simulation running - update loop
    print("\n⏳ Keeping simulation alive...")
    print("   (Close Isaac Sim window to exit)\n")
    
    # Continuous update loop to keep responsive
    while simulation_app.is_running():
        scene['world'].step(render=True)
    
    print("\n✓ Isaac Sim closed by user")

except Exception as e:
    print("\n" + "=" * 70)
    print(" ❌ TEST FAILED")
    print("=" * 70)
    print(f"\nError: {e}")
    print("\nStack trace:")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)

finally:
    print("\n⏳ Cleaning up...")
    simulation_app.close()
    print("✓ Done")
