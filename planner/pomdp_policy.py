"""
POMDP Planner: QMDP approximation for decision-making under uncertainty.
"""

import numpy as np
from typing import Dict, Tuple, List


class QMDPPlanner:
    """
    QMDP approximation for POMDP planning.
    Selects actions based on belief state and expected rewards.
    """
    
    def __init__(self, grid, viewpoint_selector, params: Dict):
        """
        Initialize QMDP planner.
        
        Args:
            grid: SemanticGrid instance
            viewpoint_selector: ViewpointSelector instance
            params: Planning parameters (rewards, thresholds, etc.)
        """
        self.grid = grid
        self.vp_selector = viewpoint_selector
        self.params = params
        
        # Reward parameters
        self.R_dock = params['reward_dock']
        self.R_step = params['reward_step']
        self.lambda_risk = params['lambda_risk']
        self.lambda_ig = params['lambda_ig']
        
        # Thresholds
        self.high_conf_thresh = params['high_confidence_threshold']
        self.med_conf_thresh = params['medium_confidence_threshold']
        self.low_conf_thresh = params['low_confidence_threshold']
        self.high_entropy_thresh = params['high_entropy_threshold']
        self.low_entropy_thresh = params['low_entropy_threshold']
        
        # State tracking
        self.num_rescans = 0
        self.max_rescans = params['max_rescans']
        
        print(f"QMDPPlanner initialized with:")
        print(f"  High confidence threshold: {self.high_conf_thresh}")
        print(f"  Max rescans: {self.max_rescans}")
    
    def select_action(self, robot_pose: np.ndarray, belief_map: np.ndarray,
                     risk_map: np.ndarray) -> Dict:
        """
        Select best action using QMDP approximation.
        
        Args:
            robot_pose: Current robot pose [x, y, theta]
            belief_map: Current charger belief distribution
            risk_map: Cost map for navigation safety
            
        Returns:
            action: Dict with 'type' and parameters
        """
        # Compute belief statistics
        max_belief = np.max(belief_map)
        entropy = self._compute_entropy(belief_map)
        
        print(f"\n--- QMDP Decision ---")
        print(f"Max belief: {max_belief:.3f}, Entropy: {entropy:.3f}, Rescans: {self.num_rescans}")
        
        # Decision logic based on belief and entropy
        if max_belief > self.high_conf_thresh and entropy < self.low_entropy_thresh:
            # High confidence, low uncertainty: approach and dock
            action = self._plan_approach(robot_pose, belief_map)
            print(f"Decision: APPROACH (high confidence)")
            
        elif max_belief > self.med_conf_thresh and self.num_rescans < self.max_rescans:
            # Medium confidence: gather more information
            action = self._plan_rescan(robot_pose, belief_map)
            print(f"Decision: RESCAN (medium confidence, gather info)")
            
        elif max_belief > self.low_conf_thresh:
            # Low-medium confidence: move to better viewpoint
            action = self._plan_move_to_viewpoint(robot_pose, belief_map)
            print(f"Decision: MOVE TO VIEWPOINT (improve visibility)")
            
        else:
            # Very low confidence: explore
            action = self._plan_exploration(robot_pose, belief_map)
            print(f"Decision: EXPLORE (low confidence)")
        
        return action
    
    def _plan_approach(self, robot_pose: np.ndarray, belief_map: np.ndarray) -> Dict:
        """
        Plan approach to highest-belief cell.
        
        Returns:
            action: {'type': 'approach', 'target': (x, y, theta), 'belief': float}
        """
        # Find cell with maximum belief
        max_idx = np.argmax(belief_map)
        j = max_idx // self.grid.grid_width
        i = max_idx % self.grid.grid_width
        
        target_world = self.grid.grid_to_world(i, j)
        
        # Compute approach pose (1m in front of target, facing it)
        # Assume target is on wall, approach from current position
        approach_distance = 1.0
        
        # Vector from target to robot
        dx = robot_pose[0] - target_world[0]
        dy = robot_pose[1] - target_world[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0.1:
            # Normalize and scale
            approach_x = target_world[0] + (dx / dist) * approach_distance
            approach_y = target_world[1] + (dy / dist) * approach_distance
        else:
            # Robot very close to target
            approach_x = target_world[0] - 1.0
            approach_y = target_world[1]
        
        # Face towards target
        approach_theta = np.arctan2(
            target_world[1] - approach_y,
            target_world[0] - approach_x
        )
        
        return {
            'type': 'approach',
            'target': (approach_x, approach_y, approach_theta),
            'target_cell': (i, j),
            'belief': float(np.max(belief_map))
        }
    
    def _plan_rescan(self, robot_pose: np.ndarray, belief_map: np.ndarray) -> Dict:
        """
        Stop and gather more observations at current location.
        
        Returns:
            action: {'type': 'rescan', 'count': int}
        """
        self.num_rescans += 1
        
        return {
            'type': 'rescan',
            'count': self.num_rescans,
            'duration': 1.0  # seconds to pause and observe
        }
    
    def _plan_move_to_viewpoint(self, robot_pose: np.ndarray, 
                               belief_map: np.ndarray) -> Dict:
        """
        Move to viewpoint with highest information gain.
        
        Returns:
            action: {'type': 'move', 'target': (x, y, theta), 'expected_ig': float}
        """
        # Generate candidate viewpoints
        candidates = self.vp_selector.generate_candidate_viewpoints(
            robot_pose, belief_map
        )
        
        # Rank by information gain
        ranked = self.vp_selector.rank_viewpoints(candidates, belief_map)
        
        # Select top viewpoint
        if len(ranked) > 0:
            best_vp, best_ig = ranked[0]
            
            return {
                'type': 'move',
                'target': best_vp,
                'expected_ig': float(best_ig)
            }
        else:
            # No good viewpoints, rotate in place
            return {
                'type': 'rotate',
                'delta_theta': np.pi / 4  # 45 degrees
            }
    
    def _plan_exploration(self, robot_pose: np.ndarray, 
                         belief_map: np.ndarray) -> Dict:
        """
        Exploration strategy when belief is very uncertain.
        
        Returns:
            action: Move to unexplored region or rotate
        """
        # Simple strategy: move towards center of room if near edge
        room_center = np.array([0.0, 0.0])
        
        distance_to_center = np.linalg.norm(robot_pose[:2] - room_center)
        
        if distance_to_center > 2.0:
            # Move towards center
            return {
                'type': 'move',
                'target': (room_center[0], room_center[1], 0.0)
            }
        else:
            # Rotate to scan more area
            return {
                'type': 'rotate',
                'delta_theta': np.pi / 2  # 90 degrees
            }
    
    def _compute_entropy(self, belief_map: np.ndarray) -> float:
        """Compute entropy of belief distribution."""
        entropy = -belief_map * np.log(belief_map + 1e-10)
        return np.sum(entropy)
    
    def compute_reward(self, state: Tuple, action: Dict, 
                      next_state: Tuple, risk: float) -> float:
        """
        Compute reward for state-action pair.
        
        R = R_dock * δ(docked) - R_step - λ_risk * risk + λ_ig * IG
        
        Args:
            state: (robot_pose, belief_map)
            action: Action dict
            next_state: (robot_pose', belief_map')
            risk: Integrated risk cost
            
        Returns:
            reward: Scalar
        """
        reward = self.R_step  # Base step cost
        
        # Docking reward
        if action['type'] == 'dock':
            reward += self.R_dock
        
        # Risk penalty
        reward -= self.lambda_risk * risk
        
        # Information gain reward
        if action['type'] in ['rescan', 'move']:
            entropy_before = self._compute_entropy(state[1])
            entropy_after = self._compute_entropy(next_state[1])
            information_gain = entropy_before - entropy_after
            reward += self.lambda_ig * information_gain
        
        return reward
    
    def reset(self):
        """Reset planner state for new episode."""
        self.num_rescans = 0


if __name__ == "__main__":
    print("Testing QMDPPlanner...")
    
    # Create dummy grid
    import sys
    sys.path.append('..')
    from mapping.semantic_grid import SemanticGrid
    from planner.viewpoint_selector import ViewpointSelector
    
    grid = SemanticGrid(
        size_meters=(6.0, 6.0),
        resolution=0.05,
        num_classes=4,
        prior={'alpha': 1.0, 'beta': 1.0, 'eta': [0.25, 0.25, 0.25, 0.25]}
    )
    
    camera_params = {'fov': 60, 'max_range': 10.0}
    vp_selector = ViewpointSelector(grid, camera_params)
    
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
    
    planner = QMDPPlanner(grid, vp_selector, planning_params)
    
    # Test with different belief scenarios
    robot_pose = np.array([0.0, 0.0, 0.0])
    
    # Scenario 1: High confidence
    belief_high = np.zeros((grid.grid_height, grid.grid_width))
    belief_high[60, 60] = 0.9
    belief_high /= np.sum(belief_high)
    
    risk_map = np.zeros((grid.grid_height, grid.grid_width))
    
    action = planner.select_action(robot_pose, belief_high, risk_map)
    assert action['type'] == 'approach'
    print(f"✓ High confidence → {action['type']}")
    
    # Scenario 2: Medium confidence
    planner.reset()
    belief_med = np.zeros((grid.grid_height, grid.grid_width))
    belief_med[60, 60] = 0.6
    belief_med /= np.sum(belief_med)
    
    action = planner.select_action(robot_pose, belief_med, risk_map)
    assert action['type'] in ['rescan', 'move']
    print(f"✓ Medium confidence → {action['type']}")
    
    print("\n✓ QMDPPlanner tests passed!")
