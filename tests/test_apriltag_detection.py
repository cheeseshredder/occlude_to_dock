"""
Test AprilTag Detection from THREE sides of charger (-X, +Y, -Y)

Uses the same viewport capture method as the working navigation code.

Run with: isaac-sim.bat --exec "test_apriltag_detection.py"
"""

import sys
import os
import numpy as np
import tempfile
import time

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("\n" + "=" * 70)
print(" APRILTAG DETECTION TEST - THREE SIDES (-X, +Y, -Y)")
print("=" * 70)

# Check if pupil_apriltags is installed
try:
    from pupil_apriltags import Detector
    print("✓ pupil_apriltags installed")
except ImportError:
    print("❌ pupil_apriltags not installed!")
    print("   Run: pip install pupil-apriltags")
    sys.exit(1)

# Start Isaac Sim
try:
    from omni.isaac.kit import SimulationApp
    print("✓ Isaac Sim available")
except ImportError as e:
    print(f"❌ Isaac Sim not available: {e}")
    sys.exit(1)

print("\n[1/4] Starting Isaac Sim...")
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
print("✓ Isaac Sim started")

try:
    # Import after SimulationApp is created
    from pxr import Gf, UsdGeom
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from PIL import Image
    import cv2
    
    print("\n[2/4] Setting up scene...")
    from scene.setup_environment import setup_scene
    scene = setup_scene('partial', seed=42, use_apriltag=True)
    world = scene['world']
    print("✓ Scene created with AprilTags on THREE sides")
    
    # Step simulation to let things settle
    for _ in range(30):
        world.step(render=True)
    
    # Setup output directory
    output_dir = os.path.join(project_root, "outputs", "apriltag_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup viewport
    viewport = get_active_viewport()
    camera_path = scene['camera_path']
    viewport.set_active_camera(camera_path)
    print(f"✓ Viewport switched to: {camera_path}")
    
    # Create detector
    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    print("\n[3/4] Testing AprilTag detection from ALL THREE sides...")
    
    # Test positions - approaching charger from different directions
    # Charger at (2.8, 0), tags on -X, +Y, -Y faces
    # Plant at (2.0, 0) blocks some views
    test_positions = [
        # Position 1: View -X face (front) from angle after going around plant
        # Robot at (2.2, 0.4), facing toward charger -X face
        # atan2(-0.4, 0.6) ≈ -34 degrees
        {"name": "-X face (front)", "pos": Gf.Vec3d(2.2, 0.4, 0.0), "rot": Gf.Vec3f(0, 0, -34)},
        
        # Position 2: View +Y face (left side) from +Y direction
        # Robot at (2.8, 1.0), facing -Y toward the +Y face of charger
        {"name": "+Y face (left side)", "pos": Gf.Vec3d(2.8, 1.0, 0.0), "rot": Gf.Vec3f(0, 0, -90)},
        
        # Position 3: View -Y face (right side) from -Y direction  
        # Robot at (2.8, -1.0), facing +Y toward the -Y face of charger
        {"name": "-Y face (right side)", "pos": Gf.Vec3d(2.8, -1.0, 0.0), "rot": Gf.Vec3f(0, 0, 90)},
    ]
    
    all_results = []
    
    for test_idx, test_pos in enumerate(test_positions):
        print(f"\n--- Test {test_idx + 1}/{len(test_positions)}: {test_pos['name']} ---")
        
        # Move robot
        robot_prim = world.stage.GetPrimAtPath("/World/Turtlebot")
        xformable = UsdGeom.Xformable(robot_prim)
        xformable.ClearXformOpOrder()
        
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(test_pos["pos"])
        
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(test_pos["rot"])
        
        print(f"  Robot at: {test_pos['pos']}, rot: {test_pos['rot']}")
        
        # Step simulation to update
        for _ in range(60):
            world.step(render=True)
        
        # Capture image with retry
        temp_path = os.path.join(tempfile.gettempdir(), f"apriltag_test_{test_idx}.png")
        
        # Try multiple times to capture
        capture_success = False
        for attempt in range(3):
            capture_viewport_to_file(viewport, temp_path)
            time.sleep(0.5 + attempt * 0.3)  # Increasing delay
            
            if os.path.exists(temp_path):
                capture_success = True
                break
            else:
                print(f"  Capture attempt {attempt + 1} failed, retrying...")
                for _ in range(30):
                    world.step(render=True)
        
        if capture_success:
            img = Image.open(temp_path)
            img = img.resize((640, 480), Image.LANCZOS)
            rgb_array = np.array(img)
            
            if rgb_array.shape[-1] == 4:
                rgb_array = rgb_array[:, :, :3]
            
            # Save capture with descriptive name
            bgr_data = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            side_name = test_pos['name'].replace(' ', '_').replace('(', '').replace(')', '')
            img_path = os.path.join(output_dir, f"capture_{side_name}.png")
            cv2.imwrite(img_path, bgr_data)
            print(f"  Saved: {img_path}")
            
            # Convert to grayscale and detect
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            
            # Camera parameters
            img_height, img_width = gray.shape[:2]
            hfov_deg = 60.0
            fx = img_width / (2 * np.tan(np.radians(hfov_deg / 2)))
            fy = fx
            cx, cy = img_width / 2, img_height / 2
            
            # Detect
            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=[fx, fy, cx, cy],
                tag_size=0.15
            )
            
            if len(detections) == 0:
                print(f"  ❌ No AprilTag detected on {test_pos['name']}")
                all_results.append({"side": test_pos['name'], "detected": False})
            else:
                det = detections[0]
                pose_t = det.pose_t.flatten() if det.pose_t is not None else [0,0,0]
                distance = np.sqrt(pose_t[0]**2 + pose_t[1]**2 + pose_t[2]**2)
                print(f"  ✅ Detected ID: {det.tag_id}, distance: {distance:.2f}m, margin: {det.decision_margin:.1f}")
                all_results.append({"side": test_pos['name'], "detected": True, "id": det.tag_id, "distance": distance})
                
                # Save annotated image
                annotated = bgr_data.copy()
                corners = det.corners.astype(int)
                for j in range(4):
                    pt1 = tuple(corners[j])
                    pt2 = tuple(corners[(j+1) % 4])
                    cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
                center = tuple(det.center.astype(int))
                cv2.circle(annotated, center, 5, (0, 0, 255), -1)
                cv2.putText(annotated, f"ID:{det.tag_id} d={distance:.2f}m", 
                           (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                annotated_path = os.path.join(output_dir, f"detection_{side_name}.png")
                cv2.imwrite(annotated_path, annotated)
            
            os.remove(temp_path)
        else:
            print(f"  ❌ Failed to capture image")
            all_results.append({"side": test_pos['name'], "detected": False})
    
    # Summary
    print(f"\n{'='*60}")
    print(f" DETECTION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for result in all_results:
        if result["detected"]:
            print(f"  ✅ {result['side']}: ID={result['id']}, dist={result['distance']:.2f}m")
        else:
            print(f"  ❌ {result['side']}: NOT DETECTED")
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL THREE SIDES DETECTED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  Some sides failed detection")
    
    print(f"{'='*60}")
    
    print("\n[4/4] Keeping simulation open for inspection...")
    print("     Close Isaac Sim window when done.\n")
    
    while simulation_app.is_running():
        world.step(render=True)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\nCleaning up...")
    simulation_app.close()
    print("Done.")
