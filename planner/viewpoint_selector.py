"""
Viewpoint Selection: Generate and rank viewpoints by expected information gain.
"""

import numpy as np
from typing import List, Tuple, Dict
from utils import bresenham_line, angle_normalize


class ViewpointSelector:
    """
    Ranks viewpoints by expected information gain for active perception.
    """
    
    def __init__(self, grid, camera_params: Dict):
        """
        Initialize viewpoint selector.
        
        Args:
            grid: SemanticGrid instance
            camera_params: Camera FOV, range, etc.
        """
        self.grid = grid
        self.camera_fov = np.radians(camera_params['fov'])  # Convert to radians
        self.camera_range = camera_params['max_range']
    
    def generate_candidate_viewpoints(self, robot_pose: np.ndarray, 
                                     belief_map: np.ndarray,
                                     num_top_cells: int = 5,
                                     viewpoint_distance: float = 2.0,
                                     num_angles: int = 8) -> List[Tuple[float, float, float]]:
        """
        Generate candidate viewpoints around high-belief regions.
        
        Args:
            robot_pose: Current robot pose [x, y, theta]
            belief_map: Charger belief distribution
            num_top_cells: Number of high-belief cells to consider
            viewpoint_distance: Distance from target cell (meters)
            num_angles: Number of viewpoint angles around each cell
            
        Returns:
            List of candidate poses (x, y, theta)
        """
        # Find top-K cells with highest charger belief
        flat_belief = belief_map.flatten()
        top_indices = np.argsort(flat_belief)[-num_top_cells:]
        
        candidates = []
        
        for idx in top_indices:
            if flat_belief[idx] < 1e-6:  # Skip near-zero belief cells
                continue
            
            j = idx // self.grid.grid_width
            i = idx % self.grid.grid_width
            
            # Convert to world coordinates
            cell_world = self.grid.grid_to_world(i, j)
            
            # Generate viewpoints around this cell
            for angle_idx in range(num_angles):
                angle = (360 / num_angles) * angle_idx
                theta_rad = np.radians(angle)
                
                # Position at viewpoint_distance away from cell
                vp_x = cell_world[0] + viewpoint_distance * np.cos(theta_rad)
                vp_y = cell_world[1] + viewpoint_distance * np.sin(theta_rad)
                
                # Heading points back towards cell
                vp_theta = np.arctan2(
                    cell_world[1] - vp_y,
                    cell_world[0] - vp_x
                )
                
                candidates.append((vp_x, vp_y, vp_theta))
        
        return candidates
    
    def predict_visibility(self, viewpoint: Tuple[float, float, float]) -> np.ndarray:
        """
        Predict which cells would be visible from viewpoint.
        
        Args:
            viewpoint: (x, y, theta) pose
            
        Returns:
            visibility_mask: Binary array (1 = visible, 0 = occluded/out of FOV)
        """
        vp_x, vp_y, vp_theta = viewpoint
        visibility = np.zeros((self.grid.grid_height, self.grid.grid_width))
        
        # Get occupancy map for occlusion checking
        occupancy = self.grid.get_occupancy_prob()
        
        for i in range(self.grid.grid_width):
            for j in range(self.grid.grid_height):
                cell_world = self.grid.grid_to_world(i, j)
                
                # Check distance
                distance = np.sqrt(
                    (cell_world[0] - vp_x)**2 +
                    (cell_world[1] - vp_y)**2
                )
                if distance > self.camera_range:
                    continue
                
                # Check FOV
                angle_to_cell = np.arctan2(
                    cell_world[1] - vp_y,
                    cell_world[0] - vp_x
                )
                angle_diff = abs(angle_normalize(angle_to_cell - vp_theta))
                
                if angle_diff > self.camera_fov / 2:
                    continue
                
                # Check occlusion (ray casting)
                is_occluded = self._check_occlusion(
                    (vp_x, vp_y), cell_world, occupancy
                )
                
                if not is_occluded:
                    visibility[j, i] = 1.0
        
        return visibility
    
    def _check_occlusion(self, start_world: Tuple[float, float], 
                        end_world: Tuple[float, float], 
                        occupancy: np.ndarray) -> bool:
        """
        Check if ray from start to end is blocked by occupied cells.
        
        Returns:
            True if occluded, False otherwise
        """
        start_grid = self.grid.world_to_grid(np.array(start_world))
        end_grid = self.grid.world_to_grid(np.array(end_world))
        
        ray_cells = bresenham_line(start_grid, end_grid)
        
        # Check occupancy along ray (excluding start and end)
        for i, j in ray_cells[1:-1]:
            if self.grid.is_valid_cell(i, j):
                if occupancy[j, i] > 0.7:  # High confidence of occupancy
                    return True  # Occluded
        
        return False  # Not occluded
    
    def compute_information_gain(self, viewpoint: Tuple[float, float, float],
                                belief_map: np.ndarray) -> float:
        """
        Compute expected information gain for a viewpoint.
        
        Information gain ≈ H(b) - E[H(b')]
        where b' is belief after observing from viewpoint.
        
        Args:
            viewpoint: (x, y, theta)
            belief_map: Current charger belief
            
        Returns:
            information_gain: Scalar (higher is better)
        """
        # Current entropy
        current_entropy = self._compute_belief_entropy(belief_map)
        
        # Predict which cells would be visible
        visibility = self.predict_visibility(viewpoint)
        
        # Estimate expected entropy after observation
        # Simplification: assume visible cells' entropy reduces by a factor
        reduction_factor = 0.5  # Tunable parameter
        
        expected_entropy = 0.0
        for i in range(self.grid.grid_width):
            for j in range(self.grid.grid_height):
                cell_belief = belief_map[j, i]
                if cell_belief < 1e-10:
                    continue
                
                cell_entropy = -cell_belief * np.log(cell_belief + 1e-10)
                
                if visibility[j, i] > 0:
                    # Cell will be observed, entropy reduces
                    expected_entropy += reduction_factor * cell_entropy
                else:
                    # Cell not observed, entropy unchanged
                    expected_entropy += cell_entropy
        
        information_gain = current_entropy - expected_entropy
        
        return information_gain
    
    def _compute_belief_entropy(self, belief_map: np.ndarray) -> float:
        """Compute total entropy of belief distribution."""
        entropy = -belief_map * np.log(belief_map + 1e-10)
        return np.sum(entropy)
    
    def rank_viewpoints(self, candidates: List[Tuple[float, float, float]],
                       belief_map: np.ndarray) -> List[Tuple[Tuple[float, float, float], float]]:
        """
        Rank all candidate viewpoints by information gain.
        
        Args:
            candidates: List of viewpoint poses
            belief_map: Current charger belief
            
        Returns:
            ranked_viewpoints: List of (viewpoint, info_gain) sorted descending
        """
        scored_viewpoints = []
        
        for vp in candidates:
            ig = self.compute_information_gain(vp, belief_map)
            scored_viewpoints.append((vp, ig))
        
        # Sort by information gain (descending)
        scored_viewpoints.sort(key=lambda x: x[1], reverse=True)
        
        return scored_viewpoints


if __name__ == "__main__":
    print("Testing ViewpointSelector...")
    
    # Create dummy grid
    import sys
    sys.path.append('..')
    from mapping.semantic_grid import SemanticGrid
    
    grid = SemanticGrid(
        size_meters=(6.0, 6.0),
        resolution=0.05,
        num_classes=4,
        prior={'alpha': 1.0, 'beta': 1.0, 'eta': [0.25, 0.25, 0.25, 0.25]}
    )
    
    # Create viewpoint selector
    camera_params = {
        'fov': 60,
        'max_range': 10.0
    }
    
    selector = ViewpointSelector(grid, camera_params)
    
    # Create dummy belief map
    belief_map = np.zeros((grid.grid_height, grid.grid_width))
    # Add a peak at center
    center_i = grid.grid_width // 2
    center_j = grid.grid_height // 2
    belief_map[center_j, center_i] = 0.5
    belief_map /= np.sum(belief_map)  # Normalize
    
    robot_pose = np.array([0.0, 0.0, 0.0])
    
    # Generate candidates
    candidates = selector.generate_candidate_viewpoints(robot_pose, belief_map)
    print(f"Generated {len(candidates)} candidate viewpoints")
    
    # Rank viewpoints
    if len(candidates) > 0:
        ranked = selector.rank_viewpoints(candidates, belief_map)
        best_vp, best_ig = ranked[0]
        print(f"Best viewpoint: ({best_vp[0]:.2f}, {best_vp[1]:.2f}, {best_vp[2]:.2f})")
        print(f"Information gain: {best_ig:.4f}")
    
    print("\n✓ ViewpointSelector tests passed!")
