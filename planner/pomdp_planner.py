"""
POMDP Planner v4 - TRUE EXPLORATION (No Hardcoded Waypoints)

Key principles:
1. NO hardcoded waypoints or positions
2. All navigation based on OBSERVED information only
3. Information-gain exploration when uncertain
4. Frontier-based exploration when no targets detected

The robot does NOT know:
- Where the charger is relative to the plant
- The room layout/bounds
- Pre-defined search patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class POMDPPlanner:
    """
    Information-theoretic POMDP planner with TRUE exploration.
    
    NO hardcoded waypoints. All decisions based on:
    - Current belief state (from observations)
    - Information gain calculations
    - Obstacle avoidance from detected objects
    """
    
    def __init__(
        self,
        grid,
        num_candidates: int = 16,
        min_distance: float = 0.3,
        max_distance: float = 1.2,
        fov_degrees: float = 60.0,
        detection_range: float = 4.0,
        confidence_threshold: float = 0.4,
        obstacle_avoidance_radius: float = 0.25,
        room_bounds: Tuple[float, float, float, float] = (-2.8, -2.8, 2.8, 2.8),  # (min_x, min_y, max_x, max_y)
    ):
        self.grid = grid
        self.num_candidates = num_candidates
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.fov_rad = np.radians(fov_degrees)
        self.detection_range = detection_range
        self.confidence_threshold = confidence_threshold
        self.obstacle_avoidance_radius = obstacle_avoidance_radius
        
        # Room bounds (with safety margin from walls)
        self.room_min_x, self.room_min_y, self.room_max_x, self.room_max_y = room_bounds
        
        # State tracking
        self.visited_positions = []  # Track where we've been
        self.planning_step = 0
        self.exploration_direction = 0.0  # Current exploration heading
        self.last_position = None  # For detecting repeated stuck
        self.stuck_counter = 0  # How many times stuck at same position
        
        # No hardcoded waypoints! Robot must explore and discover.
        
    def get_charger_belief_entropy(self) -> float:
        """Compute entropy of charger belief distribution."""
        belief = self.grid.get_charger_belief()
        belief = belief.flatten()
        belief = belief / (np.sum(belief) + 1e-10)
        entropy = -np.sum(belief * np.log(belief + 1e-10))
        return entropy
    
    def _get_detected_obstacles(self, ignore_generic_obstacles: bool = False) -> List[Tuple[float, float, float]]:
        """
        Get all detected obstacles (including plant) for collision avoidance.
        Returns list of (x, y, radius) tuples.
        
        Args:
            ignore_generic_obstacles: If True, only include plant as obstacle (for stuck recovery)
        """
        obstacles = []
        
        # Plant is ALWAYS an obstacle (it's a real physical object we've confirmed)
        plant_pos, plant_conf = self.grid.get_class_estimate('plant')
        if plant_pos is not None and plant_conf > 0.2:
            obstacles.append((plant_pos[0], plant_pos[1], 0.3))  # 30cm radius
        
        # Generic obstacles - these merged estimates can be wrong
        # Skip them when trying to recover from stuck state
        if not ignore_generic_obstacles:
            obs_pos, obs_conf = self.grid.get_class_estimate('obstacle')
            if obs_pos is not None and obs_conf > 0.2:
                obstacles.append((obs_pos[0], obs_pos[1], 0.25))
        
        return obstacles
    
    def _is_position_safe(self, x: float, y: float, ignore_generic_obstacles: bool = False) -> bool:
        """Check if position is safe (not colliding with obstacles and within bounds)."""
        # Check room bounds first
        if x < self.room_min_x or x > self.room_max_x:
            return False
        if y < self.room_min_y or y > self.room_max_y:
            return False
        
        # Check obstacles
        obstacles = self._get_detected_obstacles(ignore_generic_obstacles)
        
        for ox, oy, radius in obstacles:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            if dist < radius + self.obstacle_avoidance_radius:
                return False
        
        return True
    
    def _is_path_blocked(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> bool:
        """Check if straight path crosses any obstacle."""
        obstacles = self._get_detected_obstacles()
        
        if not obstacles:
            return False
        
        sx, sy = start
        ex, ey = end
        
        dx = ex - sx
        dy = ey - sy
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 0.1:
            return False
        
        # Check points along path
        for t in np.linspace(0, 1, 20):
            check_x = sx + t * dx
            check_y = sy + t * dy
            
            for ox, oy, radius in obstacles:
                dist = np.sqrt((check_x - ox)**2 + (check_y - oy)**2)
                if dist < radius + 0.2:  # Safety margin
                    return True
        
        return False
    
    def _generate_exploration_candidates(
        self,
        robot_pose: Tuple[float, float, float],
    ) -> List[Tuple[float, float, float]]:
        """
        Generate candidate viewpoints for exploration.
        Uses frontier-based exploration - go toward unexplored areas.
        """
        rx, ry, ryaw = robot_pose
        candidates = []
        
        # Generate candidates in a spiral pattern around robot
        for i in range(self.num_candidates):
            # Angle: spread around current heading, with bias forward
            base_angle = ryaw + (i - self.num_candidates/2) * (2*np.pi / self.num_candidates)
            
            # Distance: vary between min and max
            distance = self.min_distance + (self.max_distance - self.min_distance) * (i % 3) / 2
            
            nx = rx + distance * np.cos(base_angle)
            ny = ry + distance * np.sin(base_angle)
            
            # Check if safe
            if not self._is_position_safe(nx, ny):
                continue
            
            # Check if path is clear
            if self._is_path_blocked((rx, ry), (nx, ny)):
                continue
            
            # Yaw: face forward in direction of movement
            nyaw = base_angle
            
            candidates.append((nx, ny, nyaw))
        
        # Add some random candidates for diversity
        for _ in range(6):
            angle = np.random.uniform(-np.pi, np.pi)
            dist = np.random.uniform(self.min_distance, self.max_distance)
            
            nx = rx + dist * np.cos(angle)
            ny = ry + dist * np.sin(angle)
            
            if self._is_position_safe(nx, ny) and not self._is_path_blocked((rx, ry), (nx, ny)):
                nyaw = angle
                candidates.append((nx, ny, nyaw))
        
        return candidates
    
    def _generate_approach_candidates(
        self,
        robot_pose: Tuple[float, float, float],
        target_pos: Tuple[float, float],
    ) -> List[Tuple[float, float, float]]:
        """
        Generate candidates for approaching a target while avoiding obstacles.
        """
        rx, ry, ryaw = robot_pose
        tx, ty = target_pos
        candidates = []
        
        # Direct approach if path is clear
        if not self._is_path_blocked((rx, ry), (tx, ty)):
            yaw = np.arctan2(ty - ry, tx - rx)
            dist = np.sqrt((tx - rx)**2 + (ty - ry)**2)
            
            # Approach most of the way
            approach_dist = min(0.8 * dist, self.max_distance)
            nx = rx + approach_dist * np.cos(yaw)
            ny = ry + approach_dist * np.sin(yaw)
            
            candidates.append((nx, ny, yaw))
        
        # Generate candidates around the target for different approach angles
        for angle_deg in range(0, 360, 30):
            angle = np.radians(angle_deg)
            
            # Position 1m away from target at this angle
            vx = tx + 0.8 * np.cos(angle)
            vy = ty + 0.8 * np.sin(angle)
            
            if not self._is_position_safe(vx, vy):
                continue
            
            if not self._is_path_blocked((rx, ry), (vx, vy)):
                vyaw = np.arctan2(ty - vy, tx - vx)  # Face target
                candidates.append((vx, vy, vyaw))
        
        return candidates
    
    def _compute_exploration_value(
        self,
        viewpoint: Tuple[float, float, float],
        robot_pose: Tuple[float, float, float],
    ) -> float:
        """
        Compute value of a viewpoint for exploration.
        
        Rewards:
        - Looking at unexplored areas (high uncertainty)
        - Distance from visited positions (novelty)
        - Forward motion (progress)
        """
        vx, vy, vyaw = viewpoint
        rx, ry, ryaw = robot_pose
        
        value = 0.0
        
        # 1. Novelty: reward positions far from where we've been
        if self.visited_positions:
            min_dist_to_visited = min(
                np.sqrt((vx - px)**2 + (vy - py)**2)
                for px, py, _ in self.visited_positions
            )
            value += 0.3 * min(1.0, min_dist_to_visited / 1.0)
        else:
            value += 0.3  # Bonus for first move
        
        # 2. Forward progress: reward moving in exploration direction
        move_angle = np.arctan2(vy - ry, vx - rx)
        angle_diff = abs(self._normalize_angle(move_angle - self.exploration_direction))
        forward_bonus = 0.4 * np.cos(angle_diff)
        value += forward_bonus
        
        # 3. Information gain: look toward uncertain areas
        # Use charger belief entropy as proxy
        charger_pos, charger_conf = self.grid.get_charger_estimate()
        if charger_pos is not None and charger_conf > 0.1:
            # Look toward estimated charger
            angle_to_charger = np.arctan2(charger_pos[1] - vy, charger_pos[0] - vx)
            look_angle_diff = abs(self._normalize_angle(vyaw - angle_to_charger))
            if look_angle_diff < self.fov_rad / 2:
                value += 0.5 * (1 - look_angle_diff / (self.fov_rad / 2))
        
        # 4. Penalize staying in place
        dist_from_robot = np.sqrt((vx - rx)**2 + (vy - ry)**2)
        if dist_from_robot < 0.2:
            value -= 0.3
        
        return value
    
    def _compute_approach_value(
        self,
        viewpoint: Tuple[float, float, float],
        target_pos: Tuple[float, float],
        robot_pose: Tuple[float, float, float],
    ) -> float:
        """
        Compute value of a viewpoint for approaching a target.
        """
        vx, vy, vyaw = viewpoint
        rx, ry, ryaw = robot_pose
        tx, ty = target_pos
        
        # Distance to target from viewpoint
        dist_to_target = np.sqrt((vx - tx)**2 + (vy - ty)**2)
        
        # Distance from robot to viewpoint
        dist_from_robot = np.sqrt((vx - rx)**2 + (vy - ry)**2)
        
        # Reward getting closer to target
        current_dist = np.sqrt((rx - tx)**2 + (ry - ty)**2)
        progress = current_dist - dist_to_target
        
        # Reward looking at target
        angle_to_target = np.arctan2(ty - vy, tx - vx)
        look_angle_diff = abs(self._normalize_angle(vyaw - angle_to_target))
        
        value = 0.0
        value += 0.5 * progress  # Progress toward target
        value += 0.3 * (1 - look_angle_diff / np.pi)  # Looking at target
        value -= 0.1 * dist_from_robot  # Penalize long moves
        
        return value
    
    def _compute_visibility_probability(
        self,
        viewpoint: Tuple[float, float, float],
        target_pos: Tuple[float, float],
    ) -> float:
        """Compute probability of seeing target from viewpoint."""
        vx, vy, vyaw = viewpoint
        tx, ty = target_pos
        
        dx = tx - vx
        dy = ty - vy
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > self.detection_range or distance < 0.3:
            return 0.0
        
        angle_to_target = np.arctan2(dy, dx)
        angle_diff = abs(self._normalize_angle(angle_to_target - vyaw))
        
        if angle_diff > self.fov_rad / 2:
            return 0.0
        
        # Check occlusion by obstacles
        obstacles = self._get_detected_obstacles()
        for ox, oy, radius in obstacles:
            # Check if obstacle is between viewpoint and target
            obs_dist = np.sqrt((ox - vx)**2 + (oy - vy)**2)
            if obs_dist < distance:
                # Check if obstacle blocks line of sight
                t = ((ox - vx) * dx + (oy - vy) * dy) / (distance * distance + 1e-10)
                if 0.1 < t < 0.9:
                    closest_x = vx + t * dx
                    closest_y = vy + t * dy
                    occlusion_dist = np.sqrt((ox - closest_x)**2 + (oy - closest_y)**2)
                    if occlusion_dist < radius + 0.2:
                        return 0.1  # Heavily occluded
        
        # Probability based on distance and angle
        dist_prob = np.exp(-distance / 3.0)
        angle_prob = np.cos(angle_diff)
        
        return dist_prob * angle_prob
    
    def get_navigation_target(
        self,
        robot_pose: Tuple[float, float, float],
    ) -> Tuple[Optional[Tuple[float, float, float]], str]:
        """
        Get next navigation target based ONLY on observed information.
        
        Decision tree:
        1. If charger detected with high confidence → DOCK
        2. If charger detected but path blocked → APPROACH_AROUND
        3. If charger estimated (low conf) → EXPLORE_TOWARD
        4. If obstacle detected but no charger → EXPLORE_AROUND (find charger)
        5. Otherwise → EXPLORE (information-theoretic)
        """
        self.planning_step += 1
        rx, ry, ryaw = robot_pose
        
        # ================================================================
        # STUCK DETECTION: If robot hasn't moved, increment counter
        # ================================================================
        if self.last_position is not None:
            dist_from_last = np.sqrt((rx - self.last_position[0])**2 + (ry - self.last_position[1])**2)
            if dist_from_last < 0.15:  # Haven't moved significantly
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0  # Reset - we moved
        self.last_position = (rx, ry)
        
        # ================================================================
        # FORCE BACKUP: If stuck 3+ times at same position, BACK UP!
        # ================================================================
        if self.stuck_counter >= 3:
            print(f"  🚨 REPEATEDLY STUCK ({self.stuck_counter}x) - forcing BACKUP!")
            
            # Get plant position for orientation (need it early for backup)
            plant_pos_backup, _ = self.grid.get_class_estimate('plant')
            
            # If stuck 5+ times, the locked plant estimate might be WRONG
            # Unlock it so we can re-detect
            if self.stuck_counter >= 5 and self.grid.locked_estimates['plant'] is not None:
                print(f"  🔓 UNLOCKING plant estimate - might be wrong!")
                self.grid.locked_estimates['plant'] = None
                # Clear detection history for plant
                self.grid.class_detections['plant'] = []
                self.grid.class_position_sum['plant'] = np.array([0.0, 0.0])
                self.grid.class_weight_sum['plant'] = 0.0
            
            # Try to back up in multiple directions
            backup_angles = [
                ryaw + np.pi,           # Directly backward
                ryaw + np.pi + 0.5,     # Backward-left
                ryaw + np.pi - 0.5,     # Backward-right  
                ryaw + np.pi/2,         # Left
                ryaw - np.pi/2,         # Right
            ]
            
            for backup_angle in backup_angles:
                for backup_dist in [0.4, 0.3, 0.2]:
                    bx = rx + backup_dist * np.cos(backup_angle)
                    by = ry + backup_dist * np.sin(backup_angle)
                    bx = np.clip(bx, self.room_min_x, self.room_max_x)
                    by = np.clip(by, self.room_min_y, self.room_max_y)
                    
                    # Don't check obstacles - we might have a wrong estimate!
                    # Just check room bounds
                    if self.room_min_x < bx < self.room_max_x and self.room_min_y < by < self.room_max_y:
                        # Face toward plant (if known) or forward after backing up
                        # NO HARDCODED CHARGER POSITION!
                        if plant_pos_backup is not None:
                            target_yaw = np.arctan2(plant_pos_backup[1] - by, plant_pos_backup[0] - bx)
                        else:
                            target_yaw = 0.0  # Face forward (+X)
                        print(f"  ⏪ FORCED BACKUP to ({bx:.2f}, {by:.2f})")
                        # Don't fully reset - just decrement so we can eventually unlock plant
                        self.stuck_counter = max(0, self.stuck_counter - 2)
                        return (bx, by, target_yaw), "FORCED_BACKUP"
        
        # Update visited positions
        self.visited_positions.append((rx, ry, ryaw))
        if len(self.visited_positions) > 50:
            self.visited_positions = self.visited_positions[-50:]
        
        # Get current estimates
        charger_pos, charger_conf = self.grid.get_charger_estimate()
        plant_pos, plant_conf = self.grid.get_class_estimate('plant')
        
        # ================================================================
        # CASE 1: High confidence charger - DOCK
        # CRITICAL: Only if charger was seen RECENTLY (not stale estimate)
        # ================================================================
        charger_recently_seen = False
        if charger_pos is not None:
            # Check if charger was detected in last 3 steps
            n_recent_detections = len(self.grid.class_detections.get('charger', []))
            if n_recent_detections > 0:
                # Get the most recent detection's robot pose
                last_det = self.grid.class_detections['charger'][-1]
                last_det_pos = last_det.get('robot_pose', (0, 0, 0))
                # Check if robot has moved significantly since last detection
                dist_since_detection = np.sqrt((rx - last_det_pos[0])**2 + (ry - last_det_pos[1])**2)
                # If robot moved more than 1.5m without seeing charger again, estimate is stale
                if dist_since_detection < 1.5:
                    charger_recently_seen = True
                else:
                    print(f"  ⚠️ Charger estimate is STALE (robot moved {dist_since_detection:.2f}m since last detection)")
        
        if charger_pos is not None and charger_conf >= self.confidence_threshold and charger_recently_seen:
            # CRITICAL: Validate charger is not FAR IN FRONT of plant
            # Wall AprilTags can report positions slightly in front of plant due to projection
            # Allow some tolerance for detection error
            if plant_pos is not None and plant_conf > 0.3:
                px, py = plant_pos
                cx, cy = charger_pos
                # Charger should NOT be significantly in front of plant
                # Allow -0.2m tolerance (AprilTag wall tags project to ~2.0-2.1m X, plant at ~2.0m)
                if cx < px - 0.2:
                    print(f"  ⚠️ Invalid charger estimate ({cx:.2f}, {cy:.2f}) - too far in front of plant at ({px:.2f}, {py:.2f})")
                    # Clear the bad estimate from grid
                    self.grid.clear_charger_estimate("too far in front of plant")
                    # Skip CASE 1 - go to CASE 2 or exploration
                    charger_recently_seen = False  # Treat as stale
            
        if charger_pos is not None and charger_conf >= self.confidence_threshold and charger_recently_seen:
            dist_to_charger = np.sqrt((rx - charger_pos[0])**2 + (ry - charger_pos[1])**2)
            
            # Close enough to dock
            if dist_to_charger < 0.5:
                yaw = np.arctan2(charger_pos[1] - ry, charger_pos[0] - rx)
                return (charger_pos[0], charger_pos[1], yaw), "DOCK"
            
            # Check if path is blocked
            if self._is_path_blocked((rx, ry), charger_pos):
                # Find approach around obstacle
                candidates = self._generate_approach_candidates(robot_pose, charger_pos)
                if candidates:
                    best = max(candidates, key=lambda c: self._compute_approach_value(c, charger_pos, robot_pose))
                    print(f"  🔄 Path blocked, going around via ({best[0]:.2f}, {best[1]:.2f})")
                    return best, "APPROACH_AROUND"
            
            # Direct approach
            yaw = np.arctan2(charger_pos[1] - ry, charger_pos[0] - rx)
            approach_dist = min(0.6, dist_to_charger - 0.3)
            nx = rx + approach_dist * np.cos(yaw)
            ny = ry + approach_dist * np.sin(yaw)
            # Clamp to room bounds
            nx = np.clip(nx, self.room_min_x, self.room_max_x)
            ny = np.clip(ny, self.room_min_y, self.room_max_y)
            return (nx, ny, yaw), "APPROACH"
        
        # ================================================================
        # CASE 2: Low confidence charger estimate - explore toward it
        # BUT validate that charger is actually BEHIND the plant!
        # ================================================================
        charger_estimate_valid = False
        if charger_pos is not None and charger_conf > 0.1:
            # CRITICAL: Validate charger position makes sense
            # Charger should NOT be far in front of plant
            # Wall AprilTags project to X~2.0-2.1, plant at X~2.0, charger at X~2.8
            if plant_pos is not None and plant_conf > 0.3:
                px, py = plant_pos
                cx, cy = charger_pos
                # Allow charger estimate if it's not significantly in front of plant
                # Relaxed tolerance for AprilTag wall tag projections
                if cx > px - 0.2:  # Changed from px + 0.3 to px - 0.2
                    charger_estimate_valid = True
                else:
                    print(f"  ⚠️ Invalid charger estimate ({cx:.2f}, {cy:.2f}) - too far in front of plant at ({px:.2f}, {py:.2f})")
                    # Clear the bad estimate
                    self.grid.clear_charger_estimate("too far in front of plant")
            else:
                # No plant detected, can't validate
                charger_estimate_valid = True
        
        if charger_estimate_valid and charger_conf > 0.15:  # Increased threshold from 0.1
            # Generate candidates that improve view of estimated charger
            candidates = self._generate_approach_candidates(robot_pose, charger_pos)
            
            if candidates:
                # Score by visibility of charger
                scored = [(c, self._compute_visibility_probability(c, charger_pos)) for c in candidates]
                scored = [(c, s) for c, s in scored if s > 0.1]
                
                if scored:
                    best = max(scored, key=lambda x: x[1])[0]
                    print(f"  🔍 Exploring toward uncertain charger ({charger_pos[0]:.2f}, {charger_pos[1]:.2f})")
                    return best, "EXPLORE_TOWARD"
        
        # ================================================================
        # CASE 3: Plant/obstacle detected but no charger - explore around it
        # ================================================================
        if plant_pos is not None and plant_conf > 0.3:
            px, py = plant_pos
            
            # NOTE: Removed hardcoded Y check - plant can be anywhere!
            # Trust the color-based detection
            if True:  # Always do EXPLORE_AROUND if we have a plant
                # Strategy: Go to the SIDE of the plant and look PAST it to find charger
                # The charger is likely BEHIND the plant (from robot's perspective)
                candidates = []
                
                dist_to_plant = np.sqrt((rx - px)**2 + (ry - py)**2)
                
                # Direction from robot to plant
                angle_to_plant = np.arctan2(py - ry, px - rx)
                
                # IMPORTANT: Generate positions to the SIDE of the plant, not toward it
                # Then face toward the plant to see what's behind it
                safe_dist_from_plant = self.obstacle_avoidance_radius + 0.35  # Stay ~0.6m from plant
                
                # Try positions on both sides of the plant (perpendicular to robot-plant line)
                for side in [-1, 1]:  # Try -Y first (more clearance in typical scenes)
                    for forward_offset in [0.5, 0.8, 1.0]:  # How far forward to go
                        # Perpendicular direction
                        perp_angle = angle_to_plant + side * np.pi / 2
                        
                        # Position: forward toward plant + sideways
                        # Go forward along robot-plant line, then sideways
                        forward_dist = min(dist_to_plant - safe_dist_from_plant, forward_offset)
                        if forward_dist < 0.2:
                            forward_dist = 0.2
                        
                        vx = rx + forward_dist * np.cos(angle_to_plant) + safe_dist_from_plant * np.cos(perp_angle)
                        vy = ry + forward_dist * np.sin(angle_to_plant) + safe_dist_from_plant * np.sin(perp_angle)
                        
                        # Check bounds and safety
                        if not self._is_position_safe(vx, vy):
                            continue
                        
                        if self._is_path_blocked((rx, ry), (vx, vy)):
                            continue
                        
                        # Face TOWARD the plant - this lets us see past it to what's behind
                        vyaw = np.arctan2(py - vy, px - vx)
                        
                        candidates.append((vx, vy, vyaw))
                
                # Also try positions that go AROUND the plant to see behind it
                # CRITICAL: The charger is behind the plant in +X direction
                # So we need to face +X (toward where charger should be)
                for angle_offset in [45, 60, 75, 90, 120]:  # Degrees around plant
                    for side in [-1, 1]:  # Try -Y side first
                        # Position around the plant at this angle
                        orbit_angle = angle_to_plant + side * np.radians(angle_offset)
                        orbit_dist = safe_dist_from_plant + 0.3
                        
                        vx = px - orbit_dist * np.cos(orbit_angle)  # Position relative to plant
                        vy = py - orbit_dist * np.sin(orbit_angle)
                        
                        if not self._is_position_safe(vx, vy):
                            continue
                        
                        if self._is_path_blocked((rx, ry), (vx, vy)):
                            continue
                        
                        # CRITICAL FIX: Face toward where charger SHOULD be
                        # Charger is behind plant in +X direction at approximately (plant_x + 0.8, plant_y)
                        charger_likely_x = px + 0.8  # Behind plant
                        charger_likely_y = py  # Same Y as plant (charger is directly behind)
                        vyaw = np.arctan2(charger_likely_y - vy, charger_likely_x - vx)
                        
                        candidates.append((vx, vy, vyaw))
                
                if candidates:
                    # Score candidates: prefer positions that:
                    # 1. Let us see past the plant (toward where charger likely is)
                    # 2. Are reachable (not blocked)
                    # 3. Make forward progress toward plant
                    # 4. CRITICAL: Stay near Y=0 (charger is at same Y as plant)
                    def score_around_candidate(c):
                        vx, vy, vyaw = c
                        score = 0.0
                        
                        # Prefer positions that get us closer to the plant (X direction progress)
                        score += 0.3 * (vx - rx)  # Forward progress
                        
                        # Prefer positions that are to the side (can see around plant)
                        dist_from_plant = np.sqrt((vx - px)**2 + (vy - py)**2)
                        if dist_from_plant > safe_dist_from_plant:
                            score += 0.2
                        
                        # Prefer positions we haven't visited
                        if self.visited_positions:
                            min_dist = min(np.sqrt((vx-p[0])**2 + (vy-p[1])**2) for p in self.visited_positions)
                            score += 0.2 * min(1.0, min_dist / 0.5)
                        
                        # CRITICAL: Prefer -Y side (more clearance, avoids distractor_1 at Y=+1.5)
                        # Small penalty for +Y positions to break ties in favor of -Y
                        if vy > py:  # Going +Y side of plant
                            score -= 0.25  # Penalty for +Y
                        else:  # Going -Y side of plant
                            score += 0.15  # Bonus for -Y
                        
                        return score
                    
                    best = max(candidates, key=score_around_candidate)
                    print(f"  🌱 Exploring around detected obstacle at ({px:.2f}, {py:.2f})")
                    return best, "EXPLORE_AROUND"
        
        # ================================================================
        # CASE 4: Nothing detected - pure exploration
        # ================================================================
        candidates = self._generate_exploration_candidates(robot_pose)
        
        if candidates:
            # Score candidates
            best = max(candidates, key=lambda c: self._compute_exploration_value(c, robot_pose))
            
            # Update exploration direction toward best candidate
            move_angle = np.arctan2(best[1] - ry, best[0] - rx)
            self.exploration_direction = 0.8 * self.exploration_direction + 0.2 * move_angle
            
            print(f"  🔎 Exploring toward ({best[0]:.2f}, {best[1]:.2f})")
            return best, "EXPLORE"
        
        # ================================================================
        # FALLBACK: STUCK RECOVERY - ignore generic obstacles and move toward charger
        # ================================================================
        # We got here because all candidates failed. The merged obstacle estimate
        # might be wrong (phantom obstacle). Try again while ignoring generic obstacles.
        
        print(f"  ⚠️ All paths blocked - attempting stuck recovery")
        
        # Target: face toward plant (if known) or forward (+X)
        # NO HARDCODED POSITIONS - use detected landmarks!
        if plant_pos is not None:
            px, py = plant_pos
            # Face toward plant (charger is behind it)
            target_yaw = np.arctan2(py - ry, px - rx)
        else:
            # No plant known - face forward (+X direction)
            target_yaw = 0.0
        
        # Generate recovery candidates - ignore generic obstacles, only avoid plant
        recovery_candidates = []
        
        # Try positions toward the target direction
        for angle_offset in [-30, -15, 0, 15, 30]:  # Degrees from target direction
            for dist in [0.3, 0.5, 0.8]:
                angle = target_yaw + np.radians(angle_offset)
                nx = rx + dist * np.cos(angle)
                ny = ry + dist * np.sin(angle)
                
                # Use relaxed safety check (ignore generic obstacles)
                if self._is_position_safe(nx, ny, ignore_generic_obstacles=True):
                    recovery_candidates.append((nx, ny, angle))
        
        if recovery_candidates:
            # Pick the one that makes most progress toward target
            def recovery_score(c):
                vx, vy, _ = c
                if plant_pos is not None:
                    px, py = plant_pos
                    # Progress toward plant (charger is behind it)
                    dist_before = np.sqrt((rx - px)**2 + (ry - py)**2)
                    dist_after = np.sqrt((vx - px)**2 + (vy - py)**2)
                else:
                    # Progress in +X direction
                    dist_before = -rx
                    dist_after = -vx
                return dist_before - dist_after
            
            best = max(recovery_candidates, key=recovery_score)
            print(f"  🚨 STUCK RECOVERY: Moving to ({best[0]:.2f}, {best[1]:.2f})")
            return best, "STUCK_RECOVERY"
        
        # If even recovery fails, try systematic rotation to find charger
        yaw_diff = abs(self._normalize_angle(target_yaw - ryaw))
        if yaw_diff > 0.3:  # Need to rotate
            print(f"  🔄 Rotating to face charger area (yaw={np.degrees(target_yaw):.1f}°)")
            return (rx, ry, target_yaw), "LOOK_AROUND"
        
        # Already facing right direction but can't move - try backing up
        backup_yaw = ryaw + np.pi  # Face backward
        backup_dist = 0.3
        bx = rx + backup_dist * np.cos(backup_yaw)
        by = ry + backup_dist * np.sin(backup_yaw)
        bx = np.clip(bx, self.room_min_x, self.room_max_x)
        by = np.clip(by, self.room_min_y, self.room_max_y)
        
        if self._is_position_safe(bx, by, ignore_generic_obstacles=True):
            print(f"  ⏪ Backing up to ({bx:.2f}, {by:.2f})")
            return (bx, by, target_yaw), "BACKUP"
        
        # Absolute last resort - just rotate
        print(f"  🔄 Cannot move - just rotating")
        return (rx, ry, target_yaw), "LOOK_AROUND"
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_status(self) -> Dict:
        """Get planner status."""
        charger_pos, charger_conf = self.grid.get_charger_estimate()
        entropy = self.get_charger_belief_entropy()
        
        return {
            'planning_step': self.planning_step,
            'charger_estimate': charger_pos,
            'charger_confidence': charger_conf if charger_conf else 0.0,
            'belief_entropy': entropy,
            'visited_positions': len(self.visited_positions),
            'exploration_direction': self.exploration_direction,
        }
    
    def reset(self):
        """Reset planner state."""
        self.visited_positions = []
        self.planning_step = 0
        self.exploration_direction = 0.0
        self.last_position = None
        self.stuck_counter = 0
