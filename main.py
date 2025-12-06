"""
Main Orchestrator: Integrates all modules for the Occlude-to-Dock system.

This is the main entry point that coordinates:
- Scene setup
- Perception
- Semantic mapping
- Planning
- Control
- Evaluation
"""

import os
import sys
import time
import numpy as np
import argparse
from typing import Dict

# Import all modules
from utils import load_config, create_camera_intrinsics, create_camera_extrinsics
from perception.vision_model import PerceptionModule
from mapping.semantic_grid import SemanticGrid, create_risk_map
from planner.viewpoint_selector import ViewpointSelector
from planner.pomdp_policy import QMDPPlanner
from planner.controller import DWAController, DockingController
from evaluation.metrics_logger import EvaluationLogger


class OccludeToDockSystem:
    """
    Main system that integrates all components.
    """
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize the system with configuration files.
        
        Args:
            config_dir: Directory containing config files
        """
        print("="*60)
        print("Occlude-to-Dock System Initialization")
        print("="*60)
        
        # Load configurations
        self.params = load_config(os.path.join(config_dir, 'params.yaml'))
        self.calibration = load_config(os.path.join(config_dir, 'calibration.yaml'))
        
        print(f"\n✓ Loaded parameters from {config_dir}")
        
        # Initialize components
        self._init_perception()
        self._init_mapping()
        self._init_planning()
        self._init_control()
        
        print("\n✓ All modules initialized successfully\n")
    
    def _init_perception(self):
        """Initialize perception module."""
        print("\n[1/4] Initializing Perception...")
        
        perception_config = self.params['perception']
        temperature = self.calibration['temperature']
        
        self.perception = PerceptionModule(
            model_name=perception_config['model_name'],
            text_prompts=perception_config['text_prompts'],
            temperature=temperature
        )
        
        # Create camera parameters
        camera_config = self.params['camera']
        self.camera_params = {
            'intrinsics': create_camera_intrinsics(
                camera_config['resolution'], 
                camera_config['fov']
            ),
            'extrinsics': create_camera_extrinsics(
                camera_config['position'],
                camera_config['orientation']
            ),
            'fov': camera_config['fov'],
            'max_range': camera_config['max_range']
        }
    
    def _init_mapping(self):
        """Initialize semantic mapping."""
        print("\n[2/4] Initializing Semantic Mapping...")
        
        scene_config = self.params['scene']
        grid_config = self.params['semantic_grid']
        
        self.grid = SemanticGrid(
            size_meters=tuple(scene_config['room_size']),
            resolution=scene_config['grid_resolution'],
            num_classes=grid_config['num_classes'],
            prior={
                'alpha': grid_config['prior_alpha'],
                'beta': grid_config['prior_beta'],
                'eta': grid_config['prior_eta']
            }
        )
        
        self.inflation_radius = self.params['control']['inflation_radius']
    
    def _init_planning(self):
        """Initialize planning components."""
        print("\n[3/4] Initializing Planning...")
        
        viewpoint_config = self.params['viewpoint']
        planning_config = self.params['planning']
        
        # Viewpoint selector
        self.viewpoint_selector = ViewpointSelector(
            self.grid,
            self.camera_params
        )
        
        # QMDP planner
        self.planner = QMDPPlanner(
            self.grid,
            self.viewpoint_selector,
            planning_config
        )
    
    def _init_control(self):
        """Initialize controllers."""
        print("\n[4/4] Initializing Controllers...")
        
        control_config = self.params['control']
        docking_config = self.params['docking']
        
        self.dwa_controller = DWAController(control_config)
        
        self.docking_controller = DockingController(docking_config)
    
    def run_episode(self, scene_config: Dict, seed: int, 
                   logger: EvaluationLogger, episode_id: int,
                   max_time: float = 90.0, visualize: bool = False) -> bool:
        """
        Run a single episode.
        
        Args:
            scene_config: Scene configuration dict
            seed: Random seed
            logger: Evaluation logger
            episode_id: Episode number
            max_time: Maximum episode time in seconds
            visualize: Whether to save belief snapshots
            
        Returns:
            success: True if docked successfully
        """
        print(f"\n{'='*60}")
        print(f"Running Episode {episode_id} (seed={seed})")
        print(f"{'='*60}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Start logging
        logger.start_episode(episode_id, seed, scene_config)
        
        # Reset planner state
        self.planner.reset()
        
        # Reset grid
        self.grid.reset({
            'alpha': self.params['semantic_grid']['prior_alpha'],
            'beta': self.params['semantic_grid']['prior_beta'],
            'eta': self.params['semantic_grid']['prior_eta']
        })
        
        # TODO: Setup scene in Isaac Sim
        # For now, use mock robot interface
        robot = MockRobot(scene_config)
        
        # Get charger ground truth pose (for docking check)
        charger_pose = np.array(scene_config['objects']['charger']['position'])
        
        # Main control loop
        start_time = time.time()
        prev_pose = robot.get_pose()
        control_rate = self.params['perception']['detection_frequency']
        dt = 1.0 / control_rate
        
        step = 0
        success = False
        
        while (time.time() - start_time) < max_time:
            current_time = time.time() - start_time
            
            # Get current state
            robot_pose = robot.get_pose()
            
            # Update path metrics
            risk_map = create_risk_map(self.grid, self.inflation_radius)
            logger.update_path_metrics(prev_pose, robot_pose, risk_map, self.grid)
            prev_pose = robot_pose
            
            # Perception: Get RGB and depth frames
            rgb_frame, depth_frame = robot.get_camera_data()
            
            # Run detection
            detections = self.perception.detect(
                rgb_frame, depth_frame, robot_pose, self.camera_params
            )
            
            # Add class IDs to detections
            class_mapping = self.params['class_mapping']
            for det in detections:
                det['class_id'] = class_mapping[det['class']]
            
            # Log detections
            logger.log_detection(current_time, detections)
            
            # Update semantic grid
            self.grid.update(detections, robot_pose)
            
            # Get belief map
            belief_map = self.grid.get_charger_belief()
            
            # Log belief snapshot (every few steps)
            if visualize and step % 5 == 0:
                logger.log_belief_snapshot(current_time, belief_map)
            
            # Planning: Select action
            action = self.planner.select_action(robot_pose, belief_map, risk_map)
            logger.log_action(current_time, action)
            
            # Execute action
            if action['type'] == 'approach':
                # Check if we can dock
                if self.docking_controller.is_docked(robot_pose, charger_pose):
                    success = True
                    break
                
                # Navigate to approach pose
                target = np.array(action['target'])
                v, w = self.dwa_controller.compute_velocity(
                    robot_pose, target, risk_map, self.grid
                )
                robot.set_velocity(v, w)
                
            elif action['type'] == 'move':
                # Move to informative viewpoint
                target = np.array(action['target'])
                v, w = self.dwa_controller.compute_velocity(
                    robot_pose, target, risk_map, self.grid
                )
                robot.set_velocity(v, w)
                
            elif action['type'] == 'rotate':
                # Rotate in place
                robot.set_velocity(0.0, action['delta_theta'] / dt)
                
            elif action['type'] == 'rescan':
                # Stop and observe
                robot.set_velocity(0.0, 0.0)
            
            # Log pose
            logger.log_pose(current_time, robot_pose)
            
            # Simulate time step
            time.sleep(dt)
            step += 1
        
        # End episode
        final_pose = robot.get_pose()
        logger.end_episode(success, final_pose)
        
        return success


class MockRobot:
    """
    Mock robot interface for testing without Isaac Sim.
    Replace with actual Isaac Sim robot interface.
    """
    
    def __init__(self, scene_config: Dict):
        """Initialize mock robot."""
        self.pose = np.array(scene_config['robot_start']['position'])
        self.scene_config = scene_config
        
    def get_pose(self) -> np.ndarray:
        """Get current pose [x, y, theta]."""
        return self.pose.copy()
    
    def set_velocity(self, v: float, w: float):
        """Set velocity commands."""
        dt = 0.1
        # Update pose using differential drive kinematics
        self.pose[0] += v * np.cos(self.pose[2]) * dt
        self.pose[1] += v * np.sin(self.pose[2]) * dt
        self.pose[2] += w * dt
    
    def get_camera_data(self):
        """Get RGB and depth frames."""
        # Return dummy frames
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.uniform(0.5, 5.0, (480, 640))
        return rgb, depth


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Occlude-to-Dock System')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'experiment'],
                       help='Run mode: test (single episode) or experiment (multiple seeds)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for test mode')
    parser.add_argument('--config_dir', type=str, default='config',
                       help='Configuration directory')
    parser.add_argument('--scene', type=str, default='scene_partial.yaml',
                       help='Scene configuration file')
    parser.add_argument('--output', type=str, default='outputs/test',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Save belief snapshots')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds for experiment mode')
    
    args = parser.parse_args()
    
    # Initialize system
    system = OccludeToDockSystem(args.config_dir)
    
    # Load scene config
    scene_config = load_config(os.path.join(args.config_dir, args.scene))
    
    # Create logger
    logger = EvaluationLogger(args.output)
    
    if args.mode == 'test':
        # Single test episode
        print("\n=== TEST MODE ===\n")
        success = system.run_episode(
            scene_config, 
            args.seed, 
            logger, 
            episode_id=0,
            visualize=args.visualize
        )
        
        print(f"\n{'='*60}")
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"{'='*60}\n")
        
    elif args.mode == 'experiment':
        # Multiple seeds
        print(f"\n=== EXPERIMENT MODE ({args.num_seeds} seeds) ===\n")
        
        for episode_id in range(args.num_seeds):
            seed = 1000 + episode_id
            success = system.run_episode(
                scene_config,
                seed,
                logger,
                episode_id=episode_id,
                visualize=args.visualize
            )
        
        # Save summary
        logger.save_summary()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
