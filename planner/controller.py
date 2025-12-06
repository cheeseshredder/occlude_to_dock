"""
Controller: Local motion planning (DWA) and docking execution.
"""

import numpy as np
from typing import Dict, Tuple, List
from utils import angle_normalize


class DWAController:
    """
    Dynamic Window Approach for local motion control.
    """
    
    def __init__(self, params: Dict):
        """
        Initialize DWA controller.
        
        Args:
            params: Robot kinematics and DWA parameters
        """
        self.max_v = params['max_linear_vel']
        self.max_w = params['max_angular_vel']
        self.v_resolution = params['v_resolution']
        self.w_resolution = params['w_resolution']
        self.dt = params['dt']
        self.horizon = params['dwa_horizon']
        
        print(f"DWAController initialized:")
        print(f"  Max velocities: v={self.max_v:.2f} m/s, w={self.max_w:.2f} rad/s")
        print(f"  Horizon: {self.horizon:.1f}s")
    
    def compute_velocity(self, robot_pose: np.ndarray, target_pose: np.ndarray,
                        risk_map: np.ndarray, grid) -> Tuple[float, float]:
        """
        Compute optimal velocity using DWA.
        
        Args:
            robot_pose: Current pose [x, y, theta]
            target_pose: Goal pose [x, y, theta]
            risk_map: Cost map for collision avoidance
            grid: SemanticGrid for bounds checking
            
        Returns:
            (v_linear, v_angular) command velocities
        """
        # Dynamic window: reachable velocities
        v_min = max(-self.max_v, -0.5)
        v_max = self.max_v
        w_min = -self.max_w
        w_max = self.max_w
        
        best_score = -float('inf')
        best_v = 0.0
        best_w = 0.0
        
        # Evaluate velocity pairs
        for v in np.arange(v_min, v_max + self.v_resolution, self.v_resolution):
            for w in np.arange(w_min, w_max + self.w_resolution, self.w_resolution):
                # Simulate trajectory
                trajectory = self._simulate_trajectory(robot_pose, v, w)
                
                # Check collision
                if self._check_collision(trajectory, risk_map, grid):
                    continue
                
                # Score trajectory
                score = self._score_trajectory(trajectory, target_pose, risk_map, grid)
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
        
        # If no valid trajectory found, stop
        if best_score == -float('inf'):
            return (0.0, 0.0)
        
        return (best_v, best_w)
    
    def _simulate_trajectory(self, robot_pose: np.ndarray, v: float, 
                            w: float) -> List[Tuple[float, float, float]]:
        """
        Simulate trajectory for given velocities.
        
        Returns:
            List of poses [(x, y, theta), ...]
        """
        trajectory = []
        x, y, theta = robot_pose
        
        for _ in np.arange(0, self.horizon, self.dt):
            # Update pose (differential drive kinematics)
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            theta = angle_normalize(theta)
            
            trajectory.append((x, y, theta))
        
        return trajectory
    
    def _check_collision(self, trajectory: List[Tuple[float, float, float]],
                        risk_map: np.ndarray, grid) -> bool:
        """
        Check if trajectory collides with obstacles.
        
        Returns:
            True if collision, False otherwise
        """
        for x, y, theta in trajectory:
            # Convert to grid coordinates
            i, j = grid.world_to_grid(np.array([x, y]))
            
            # Check bounds
            if not grid.is_valid_cell(i, j):
                return True  # Out of bounds
            
            # Check risk (high risk = obstacle)
            if risk_map[j, i] > 0.8:
                return True  # Collision
        
        return False
    
    def _score_trajectory(self, trajectory: List[Tuple[float, float, float]],
                         target_pose: np.ndarray, risk_map: np.ndarray,
                         grid) -> float:
        """
        Score trajectory based on goal proximity, heading, and safety.
        
        Returns:
            score: Scalar (higher is better)
        """
        final_pose = trajectory[-1]
        
        # Distance to goal
        dist_to_goal = np.sqrt(
            (final_pose[0] - target_pose[0])**2 +
            (final_pose[1] - target_pose[1])**2
        )
        
        # Heading alignment
        target_heading = np.arctan2(
            target_pose[1] - final_pose[1],
            target_pose[0] - final_pose[0]
        )
        heading_error = abs(angle_normalize(final_pose[2] - target_heading))
        
        # Safety (average risk along trajectory)
        total_risk = 0.0
        for x, y, theta in trajectory:
            i, j = grid.world_to_grid(np.array([x, y]))
            if grid.is_valid_cell(i, j):
                total_risk += risk_map[j, i]
        avg_risk = total_risk / len(trajectory) if len(trajectory) > 0 else 0.0
        
        # Combined score
        score = (
            -1.0 * dist_to_goal +
            -0.5 * heading_error +
            -2.0 * avg_risk
        )
        
        return score


class DockingController:
    """
    Final alignment and docking maneuver.
    """
    
    def __init__(self, tolerances: Dict):
        """
        Initialize docking controller.
        
        Args:
            tolerances: Position and yaw tolerances for success
        """
        self.pos_tol = tolerances['position_tolerance']
        self.yaw_tol = tolerances['yaw_tolerance']
        self.approach_speed = tolerances['approach_speed']
        self.max_attempts = tolerances['max_attempts']
        
        print(f"DockingController initialized:")
        print(f"  Position tolerance: {self.pos_tol:.3f}m")
        print(f"  Yaw tolerance: {self.yaw_tol:.3f}rad")
    
    def is_docked(self, robot_pose: np.ndarray, 
                  charger_pose: np.ndarray) -> bool:
        """
        Check if robot is successfully docked.
        
        Args:
            robot_pose: Current robot pose [x, y, theta]
            charger_pose: Charger pose [x, y, theta]
            
        Returns:
            True if within tolerances, False otherwise
        """
        # Position error
        dx = charger_pose[0] - robot_pose[0]
        dy = charger_pose[1] - robot_pose[1]
        position_error = np.sqrt(dx**2 + dy**2)
        
        # Heading error
        heading_error = abs(angle_normalize(charger_pose[2] - robot_pose[2]))
        
        return (position_error < self.pos_tol and 
                heading_error < self.yaw_tol)
    
    def compute_docking_velocity(self, robot_pose: np.ndarray,
                                 charger_pose: np.ndarray) -> Tuple[float, float]:
        """
        Compute velocity for final docking approach.
        
        Returns:
            (v_linear, v_angular)
        """
        # Get current distance and heading error
        dx = charger_pose[0] - robot_pose[0]
        dy = charger_pose[1] - robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        target_heading = np.arctan2(dy, dx)
        heading_error = angle_normalize(target_heading - robot_pose[2])
        
        # Check if docked
        if self.is_docked(robot_pose, charger_pose):
            return (0.0, 0.0)
        
        # Control logic
        if abs(heading_error) > 0.1:  # Align first
            v_linear = 0.0
            v_angular = 0.3 * np.sign(heading_error)
        else:  # Move forward
            v_linear = min(self.approach_speed, distance)
            v_angular = 0.2 * heading_error  # Small correction
        
        return (v_linear, v_angular)


def navigate_to_pose(robot, target_pose: np.ndarray, dwa_controller: DWAController,
                     risk_map: np.ndarray, grid, max_steps: int = 100) -> bool:
    """
    Navigate robot to target pose using DWA.
    
    Args:
        robot: Robot interface
        target_pose: Target [x, y, theta]
        dwa_controller: DWA controller instance
        risk_map: Risk/cost map
        grid: Semantic grid
        max_steps: Maximum control steps
        
    Returns:
        True if reached, False if failed
    """
    for step in range(max_steps):
        # Get current pose
        robot_pose = robot.get_pose()
        
        # Check if reached
        dx = target_pose[0] - robot_pose[0]
        dy = target_pose[1] - robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.2:  # Close enough
            robot.set_velocity(0.0, 0.0)
            return True
        
        # Compute velocity
        v, w = dwa_controller.compute_velocity(robot_pose, target_pose, risk_map, grid)
        
        # Execute
        robot.set_velocity(v, w)
        
        # Small delay (in real system would be control loop rate)
        # time.sleep(0.1)
    
    # Failed to reach
    robot.set_velocity(0.0, 0.0)
    return False


if __name__ == "__main__":
    print("Testing Controllers...")
    
    # Test DWA
    params = {
        'max_linear_vel': 0.3,
        'max_angular_vel': 0.5,
        'v_resolution': 0.05,
        'w_resolution': 0.1,
        'dt': 0.1,
        'dwa_horizon': 1.0
    }
    
    dwa = DWAController(params)
    
    # Test velocity computation with dummy data
    import sys
    sys.path.append('..')
    from mapping.semantic_grid import SemanticGrid
    
    grid = SemanticGrid(
        size_meters=(6.0, 6.0),
        resolution=0.05,
        num_classes=4,
        prior={'alpha': 1.0, 'beta': 1.0, 'eta': [0.25, 0.25, 0.25, 0.25]}
    )
    
    robot_pose = np.array([0.0, 0.0, 0.0])
    target_pose = np.array([2.0, 1.0, 0.0])
    risk_map = np.zeros((grid.grid_height, grid.grid_width))
    
    v, w = dwa.compute_velocity(robot_pose, target_pose, risk_map, grid)
    print(f"✓ DWA velocity: v={v:.3f}, w={w:.3f}")
    
    # Test docking controller
    dock_params = {
        'position_tolerance': 0.15,
        'yaw_tolerance': 0.2,
        'approach_speed': 0.1,
        'max_attempts': 50
    }
    
    docker = DockingController(dock_params)
    
    # Test docking check
    charger_pose = np.array([2.0, 0.0, 0.0])
    robot_near = np.array([2.1, 0.05, 0.05])
    robot_far = np.array([3.0, 0.0, 0.0])
    
    assert docker.is_docked(robot_near, charger_pose) == True
    assert docker.is_docked(robot_far, charger_pose) == False
    print("✓ Docking checks passed")
    
    print("\n✓ Controller tests passed!")
