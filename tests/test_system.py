"""
Test Script: Verify all modules are working correctly.

Run this script to test the entire pipeline without Isaac Sim.
"""

import sys
import numpy as np

print("="*60)
print("Occlude-to-Dock Module Tests")
print("="*60)

# Test 1: Utilities
print("\n[1/7] Testing utilities...")
try:
    from utils import angle_normalize, compute_iou, bresenham_line, sigmoid
    
    assert np.isclose(angle_normalize(3.5 * np.pi), -0.5 * np.pi)
    assert np.isclose(compute_iou([0,0,10,10], [5,5,15,15]), 0.142857, atol=1e-5)
    assert np.isclose(sigmoid(0), 0.5)
    
    print("✓ Utilities working")
except Exception as e:
    print(f"✗ Utilities failed: {e}")
    sys.exit(1)

# Test 2: Semantic Grid
print("\n[2/7] Testing semantic grid...")
try:
    from mapping.semantic_grid import SemanticGrid
    
    grid = SemanticGrid(
        size_meters=(6.0, 6.0),
        resolution=0.05,
        num_classes=4,
        prior={'alpha': 1.0, 'beta': 1.0, 'eta': [0.25, 0.25, 0.25, 0.25]}
    )
    
    # Test detection update
    detection = {
        'center_3d': np.array([1.0, 0.5, 0.3]),
        'calibrated_prob': 0.85,
        'class_id': 0,
        'distance': 1.5
    }
    
    robot_pose = np.array([0.0, 0.0, 0.0])
    grid.update([detection], robot_pose)
    
    belief = grid.get_charger_belief()
    assert belief.shape == (grid.grid_height, grid.grid_width)
    
    print("✓ Semantic grid working")
except Exception as e:
    print(f"✗ Semantic grid failed: {e}")
    sys.exit(1)

# Test 3: Perception
print("\n[3/7] Testing perception module...")
try:
    from perception.vision_model import PerceptionModule
    
    text_prompts = {
        'charger': 'a wall-mounted charging dock',
        'plant': 'a potted plant'
    }
    
    perception = PerceptionModule(
        model_name="IDEA-Research/grounding-dino-base",
        text_prompts=text_prompts,
        temperature=2.0
    )
    
    # Test with mock data
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 5.0, (480, 640))
    robot_pose = np.array([0.0, 0.0, 0.0])
    camera_params = {
        'intrinsics': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        'extrinsics': np.eye(4)
    }
    
    detections = perception.detect(rgb, depth, robot_pose, camera_params)
    assert isinstance(detections, list)
    
    print("✓ Perception module working")
except Exception as e:
    print(f"✗ Perception failed: {e}")
    sys.exit(1)

# Test 4: Viewpoint Selector
print("\n[4/7] Testing viewpoint selector...")
try:
    from planner.viewpoint_selector import ViewpointSelector
    
    camera_params = {'fov': 60, 'max_range': 10.0}
    selector = ViewpointSelector(grid, camera_params)
    
    belief_map = np.zeros((grid.grid_height, grid.grid_width))
    belief_map[60, 60] = 0.5
    belief_map /= np.sum(belief_map)
    
    candidates = selector.generate_candidate_viewpoints(robot_pose, belief_map)
    assert len(candidates) > 0
    
    print("✓ Viewpoint selector working")
except Exception as e:
    print(f"✗ Viewpoint selector failed: {e}")
    sys.exit(1)

# Test 5: QMDP Planner
print("\n[5/7] Testing QMDP planner...")
try:
    from planner.pomdp_policy import QMDPPlanner
    
    planning_params = {
        'reward_dock': 100.0,
        'reward_step': -1.0,
        'lambda_risk': 0.5,
        'lambda_ig': 2.0,
        'max_rescans': 3,
        'high_confidence_threshold': 0.8,
        'medium_confidence_threshold': 0.5,
        'low_confidence_threshold': 0.3,
        'high_entropy_threshold': 1.0,
        'low_entropy_threshold': 0.5
    }
    
    planner = QMDPPlanner(grid, selector, planning_params)
    
    risk_map = np.zeros((grid.grid_height, grid.grid_width))
    action = planner.select_action(robot_pose, belief_map, risk_map)
    assert 'type' in action
    
    print("✓ QMDP planner working")
except Exception as e:
    print(f"✗ QMDP planner failed: {e}")
    sys.exit(1)

# Test 6: Controllers
print("\n[6/7] Testing controllers...")
try:
    from planner.controller import DWAController, DockingController
    
    dwa_params = {
        'max_linear_vel': 0.3,
        'max_angular_vel': 0.5,
        'v_resolution': 0.05,
        'w_resolution': 0.1,
        'dt': 0.1,
        'dwa_horizon': 1.0
    }
    
    dwa = DWAController(dwa_params)
    
    target = np.array([2.0, 1.0, 0.0])
    v, w = dwa.compute_velocity(robot_pose, target, risk_map, grid)
    assert isinstance(v, float) and isinstance(w, float)
    
    dock_params = {
        'position_tolerance': 0.15,
        'yaw_tolerance': 0.2,
        'approach_speed': 0.1,
        'max_attempts': 50
    }
    
    docker = DockingController(dock_params)
    charger_pose = np.array([2.0, 0.0, 0.0])
    is_docked = docker.is_docked(robot_pose, charger_pose)
    assert isinstance(is_docked, bool)
    
    print("✓ Controllers working")
except Exception as e:
    print(f"✗ Controllers failed: {e}")
    sys.exit(1)

# Test 7: Evaluation Logger
print("\n[7/7] Testing evaluation logger...")
try:
    from evaluation.metrics_logger import EvaluationLogger
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(tmpdir)
        
        logger.start_episode(0, 42, {'test': True})
        logger.log_action(0.5, {'type': 'move'})
        logger.end_episode(True, np.array([2.0, 0.0, 0.0]))
        
        metrics = logger.compute_aggregate_metrics()
        assert 'success_rate' in metrics
    
    print("✓ Evaluation logger working")
except Exception as e:
    print(f"✗ Evaluation logger failed: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nThe system is ready to run.")
print("\nNext steps:")
print("  1. Collect calibration data: python calibration/collect_dataset.py")
print("  2. Fit temperature: python calibration/fit_temperature.py")
print("  3. Run test episode: python main.py --mode test --visualize")
print("  4. Run full experiment: python main.py --mode experiment --num_seeds 5")
