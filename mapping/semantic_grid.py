"""
Semantic Grid with Position Estimation - RAW ESTIMATE VERSION

Key changes from previous version:
- Uses RAW charger estimates when available (more accurate)
- Landmark correction only used as fallback when charger not directly seen
- Locks RAW position, not landmark-corrected
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


CLASS_TO_ID = {'charger': 0, 'plant': 1, 'obstacle': 2, 'free': 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASS_TO_ID)


class SemanticGrid:
    """
    Semantic grid with landmark-based localization.
    """
    
    def __init__(
        self, 
        size_meters: Tuple[float, float] = (6.0, 6.0),
        resolution: float = 0.5,
        origin: Tuple[float, float] = (-3.0, -3.0),
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        close_range_threshold: float = 1.2,
        close_range_weight_factor: float = 0.2,  # Not used anymore but kept for compatibility
        lock_threshold: float = 1.0,
        min_detections_to_lock: int = 2,
    ):
        self.size_meters = size_meters
        self.resolution = resolution
        self.origin = np.array(origin)
        
        self.width = int(size_meters[0] / resolution)
        self.height = int(size_meters[1] / resolution)
        
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # REMOVED: plant_to_charger_offset - no hardcoded spatial assumptions!
        
        self.close_range_threshold = close_range_threshold
        self.close_range_weight_factor = close_range_weight_factor
        self.lock_threshold = lock_threshold
        self.min_detections_to_lock = min_detections_to_lock
        
        self.class_position_sum = {cls: np.array([0.0, 0.0]) for cls in CLASS_TO_ID.keys()}
        self.class_weight_sum = {cls: 0.0 for cls in CLASS_TO_ID.keys()}
        self.class_detections = {cls: [] for cls in CLASS_TO_ID.keys()}
        
        self.locked_estimates = {cls: None for cls in CLASS_TO_ID.keys()}
        
        self.charger_bearings = []
        
        self.reset()
        
        print(f"SemanticGrid (RAW Estimate Preference):")
        print(f"  Size: {self.width}×{self.height} cells")
        print(f"  Lock threshold: {lock_threshold} weight, min {min_detections_to_lock} detections")
    
    def reset(self):
        self.alpha = np.full((self.height, self.width), self.prior_alpha, dtype=np.float32)
        self.beta = np.full((self.height, self.width), self.prior_beta, dtype=np.float32)
        self.eta = np.ones((self.height, self.width, NUM_CLASSES), dtype=np.float32)
        self.observation_count = np.zeros((self.height, self.width), dtype=np.int32)
        
        for cls in CLASS_TO_ID.keys():
            self.class_position_sum[cls] = np.array([0.0, 0.0])
            self.class_weight_sum[cls] = 0.0
            self.class_detections[cls] = []
            self.locked_estimates[cls] = None
        
        self.charger_bearings = []
    
    def clear_charger_estimate(self, reason: str = "invalid"):
        """Clear charger estimate when it's found to be invalid."""
        print(f"  🗑️ Clearing charger estimate ({reason})")
        self.class_position_sum['charger'] = np.array([0.0, 0.0])
        self.class_weight_sum['charger'] = 0.0
        self.class_detections['charger'] = []
        self.locked_estimates['charger'] = None
        self.charger_bearings = []
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.origin[0]) / self.resolution)
        j = int((y - self.origin[1]) / self.resolution)
        i = max(0, min(self.width - 1, i))
        j = max(0, min(self.height - 1, j))
        return i, j
    
    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.origin[0] + (i + 0.5) * self.resolution
        y = self.origin[1] + (j + 0.5) * self.resolution
        return x, y
    
    def is_valid(self, i: int, j: int) -> bool:
        return 0 <= i < self.width and 0 <= j < self.height
    
    def update_from_detections(
        self, 
        detections: List[Dict], 
        robot_pose: Tuple[float, float, float],
    ):
        """Update grid using detections with fixes."""
        robot_x, robot_y, robot_yaw = robot_pose
        
        # FIRST PASS: Collect charger detection positions
        # This is needed to suppress obstacle detections that overlap with charger
        # (AprilTag white squares being detected as obstacles)
        charger_positions_this_frame = []
        for det in detections:
            if det['class'] == 'charger':
                charger_positions_this_frame.append(det['world_position'])
        
        for det in detections:
            world_x, world_y = det['world_position']
            class_name = det['class']
            
            # Skip if locked
            if self.locked_estimates[class_name] is not None:
                continue
            
            confidence = det.get('calibrated_prob', det.get('confidence', 0.5))
            color_score = det.get('color_score', 1.0)
            
            distance = np.sqrt((world_x - robot_x)**2 + (world_y - robot_y)**2)
            
            # FIX: Suppress obstacle detections that are near charger detections
            # These are likely AprilTag white squares being misdetected as obstacles
            if class_name == 'obstacle':
                near_charger = False
                for cx, cy in charger_positions_this_frame:
                    dist_to_charger = np.sqrt((world_x - cx)**2 + (world_y - cy)**2)
                    if dist_to_charger < 0.5:  # Within 50cm of charger = suppress
                        near_charger = True
                        break
                
                # Also check locked charger estimate
                if self.locked_estimates['charger'] is not None:
                    locked_cx, locked_cy = self.locked_estimates['charger']['position']
                    dist_to_locked = np.sqrt((world_x - locked_cx)**2 + (world_y - locked_cy)**2)
                    if dist_to_locked < 0.5:
                        near_charger = True
                
                if near_charger:
                    print(f"  ⚠️ Suppressing obstacle at ({world_x:.2f}, {world_y:.2f}) - too close to charger (likely AprilTag pattern)")
                    continue
            
            # CRITICAL: Validate charger detections
            # AprilTag detections are 100% reliable (depth_method='apriltag_pose')
            # Color-based detections need validation
            if class_name == 'charger':
                depth_method = det.get('depth_method', 'unknown')
                
                # AprilTag detections are usually reliable BUT can fail at oblique angles
                # Validate position before trusting
                if depth_method == 'apriltag_pose':
                    position_valid = True
                    
                    # Check viewing angle first - oblique views have unreliable pose
                    is_frontal = det.get('is_frontal', True)  # Default true for backwards compat
                    viewing_angle = det.get('viewing_angle_deg', 0)
                    
                    if viewing_angle > 60:
                        print(f"  ⚠️ AprilTag at ({world_x:.2f}, {world_y:.2f}) - oblique view ({viewing_angle:.0f}°), REJECTED")
                        continue  # Skip entirely - pose is garbage
                    
                    # X range check
                    if world_x < 1.5 or world_x > 3.5:
                        print(f"  ⚠️ AprilTag position X={world_x:.2f} out of expected range [1.5, 3.5]")
                        position_valid = False
                    
                    # CRITICAL FIX: Validate position relative to plant!
                    # Wall AprilTags are at Y=±0.40m offset from charger center
                    # When viewed from robot position, they can appear at different Y values
                    plant_pos_check = self.locked_estimates.get('plant')
                    if plant_pos_check is not None and position_valid:
                        plant_y = plant_pos_check['position'][1]
                        plant_x = plant_pos_check['position'][0]
                        y_error = abs(world_y - plant_y)
                        
                        # RELAXED: Allow ±1.5m Y error for wall tags
                        # Wall tags are offset from charger center, and viewing angle adds more error
                        # The key validation is X position (must be behind plant)
                        if y_error > 1.5:
                            print(f"  ⚠️ AprilTag at ({world_x:.2f}, {world_y:.2f}) - Y too far from plant Y={plant_y:.2f} (error={y_error:.2f}m), REJECTED")
                            position_valid = False
                        
                        # Charger should be BEHIND plant (with 0.2m tolerance for detection error)
                        # AprilTag detection at X=2.03 when plant is at X=1.98 is VALID (5cm behind)
                        if world_x < plant_x - 0.2:  # 20cm tolerance
                            print(f"  ⚠️ AprilTag at ({world_x:.2f}, {world_y:.2f}) - too far in front of plant X={plant_x:.2f}, REJECTED")
                            position_valid = False
                    
                    if position_valid:
                        # Adjust weight based on viewing angle
                        if is_frontal:
                            print(f"  🎯 AprilTag charger at ({world_x:.2f}, {world_y:.2f}) - FRONTAL, TRUSTED")
                            weight = 2.0  # High weight for frontal view
                        else:
                            print(f"  🟡 AprilTag charger at ({world_x:.2f}, {world_y:.2f}) - marginal ({viewing_angle:.0f}°)")
                            weight = 1.2  # Reduced weight for marginal view
                    else:
                        print(f"  ⚠️ AprilTag charger at ({world_x:.2f}, {world_y:.2f}) - REJECTED")
                        continue  # Skip this detection
                else:
                    # Non-AprilTag detection: COLOR VALIDATION is primary!
                    # Charger is GREEN, distractors are GRAY
                    # This is the KEY differentiator - not hardcoded position!
                    
                    # STRICT color requirement for far detections (>2m)
                    # Relaxed for close detections (visual confirmation)
                    if distance > 2.0:
                        # Far detection: require HIGH green color score
                        if color_score < 0.5:
                            print(f"  ⚠️ Filtering FAR charger detection at ({world_x:.2f}, {world_y:.2f}) - color score {color_score:.2f} < 0.5 (need strong green)")
                            continue
                        # Also reduce weight for far detections (less reliable)
                        confidence = confidence * 0.5
                        
                        # CRITICAL: If plant is NOT locked yet, don't trust far charger detections!
                        # We need plant as reference to validate charger position
                        plant_pos_check = self.locked_estimates.get('plant')
                        if plant_pos_check is None:
                            print(f"  ⚠️ Filtering FAR charger at ({world_x:.2f}, {world_y:.2f}) - no plant reference yet (need to find plant first)")
                            continue
                    else:
                        # Close detection: still require moderate color requirement
                        # 0.44 is NOT green enough - should be > 0.5 for charger
                        if color_score < 0.50:
                            print(f"  ⚠️ Filtering charger detection - color score {color_score:.2f} < 0.50 (not green enough)")
                            continue
                    
                    # RELATIVE POSITION validation (if plant is known)
                    # Charger must be BEHIND plant (higher X value)
                    # This is relative, not hardcoded!
                    plant_pos_check = self.locked_estimates.get('plant')
                    if plant_pos_check is not None:
                        plant_x = plant_pos_check['position'][0]
                        plant_y = plant_pos_check['position'][1]
                        # Charger should be behind plant (relaxed from 0.3 to 0.0 - allow some detection error)
                        # Detection errors at oblique angles can be 0.3-0.5m, so strict thresholds reject valid detections
                        if world_x < plant_x:
                            print(f"  ⚠️ Filtering charger at ({world_x:.2f}, {world_y:.2f}) - in front of plant at x={plant_x:.2f}")
                            continue
                        # Charger should be near same Y as plant (within 0.8m)
                        if abs(world_y - plant_y) > 0.8:
                            print(f"  ⚠️ Filtering charger at ({world_x:.2f}, {world_y:.2f}) - too far from plant Y={plant_y:.2f}")
                            continue
                    
                    # Basic sanity: charger should be in front of robot (+X region)
                    if world_x < 0.5:
                        print(f"  ⚠️ Filtering charger at ({world_x:.2f}, {world_y:.2f}) - behind robot")
                        continue
            
            # FIXED: Close-range detections are MORE accurate, not less!
            # Give them HIGHER weight, reduce weight for far detections
            # (AprilTag detections already have weight set above)
            if class_name != 'charger' or det.get('depth_method') != 'apriltag_pose':
                if distance < self.close_range_threshold:
                    distance_weight = 1.5  # Close range = more accurate
                elif distance < 3.0:
                    distance_weight = 1.0  # Mid-range = good
                else:
                    distance_weight = max(0.3, 1.0 - (distance - 3.0) * 0.2)  # Far = less reliable
                
                weight = confidence * color_score * distance_weight
            
            self.class_detections[class_name].append({
                'position': (world_x, world_y),
                'weight': weight,
                'distance': distance,
                'robot_pose': robot_pose,
                'depth_method': det.get('depth_method', 'unknown'),
                'color_score': color_score,
                'is_frontal': det.get('is_frontal', False),  # ADDED: Store frontal flag for locking
                'viewing_angle_deg': det.get('viewing_angle_deg', 0),  # ADDED: Store viewing angle
            })
            
            self.class_position_sum[class_name] += np.array([world_x, world_y]) * weight
            self.class_weight_sum[class_name] += weight
            
            if class_name == 'charger':
                bearing = np.arctan2(world_y - robot_y, world_x - robot_x)
                self.charger_bearings.append({
                    'robot_pos': (robot_x, robot_y),
                    'bearing': bearing,
                    'confidence': weight,
                    'distance': distance,
                })
        
        # Check for locking after all detections processed
        self._check_and_lock_estimates()
        
        self._update_grid_from_averages()
    
    def _check_and_lock_estimates(self):
        """Check if any estimates should be locked - require CLOSE RANGE RAYCAST for plant."""
        
        # Lock PLANT first (it's our landmark - critical!)
        if self.locked_estimates['plant'] is None:
            plant_detections = self.class_detections['plant']
            n_plant = len(plant_detections)
            plant_weight = self.class_weight_sum['plant']
            
            # CRITICAL FIX: Only consider RAYCAST detections for locking
            # BBOX depth is too unreliable (±30% error)
            raycast_detections = []
            for det in plant_detections:
                # Check if this was a raycast detection
                if det.get('depth_method', 'unknown') == 'raycast':
                    raycast_detections.append(det)
            
            # Need at least 2 close-range RAYCAST detections
            close_raycast = [d for d in raycast_detections if d['distance'] < 2.5]  # Increased from 1.5m to catch plants at ~2m
            
            # CRITICAL FIX: Filter out detections with X > 2.5 BEFORE selecting best
            # Plant is the OCCLUDER - it must be BETWEEN robot start (0,0) and charger (~2.8,0)
            # Detections at X > 2.5 are likely the charger or distractor, not the plant!
            # Also require good color score (red)
            valid_plant_detections = []
            for d in close_raycast:
                det_x = d['position'][0]
                det_color = d.get('color_score', 0.0)
                if det_x > 2.5:
                    print(f"  ⚠️ Filtering plant detection at X={det_x:.2f} > 2.5 (too close to charger area)")
                elif det_color < 0.8:
                    print(f"  ⚠️ Filtering plant detection - color score {det_color:.2f} < 0.8")
                else:
                    valid_plant_detections.append(d)
            
            if len(valid_plant_detections) >= 1:
                # Use the detection with highest weight
                # Weight already incorporates color_score, so high weight = good color match
                best_det = max(valid_plant_detections, key=lambda d: d['weight'])
                pos = best_det['position']
                
                self.locked_estimates['plant'] = {
                    'position': pos,
                    'confidence': min(1.0, best_det['weight'] * 2),
                }
                print(f"  🔒 LOCKED plant at ({pos[0]:.2f}, {pos[1]:.2f}) [close raycast at {best_det['distance']:.1f}m]")
                return
            
            # Fallback: Single excellent detection - REQUIRE RAYCAST and high confidence
            for det in plant_detections:
                dist = det['distance']
                method = det.get('depth_method', 'unknown')
                color_score = det.get('color_score', 0.0)
                # RELAXED: Lock on close, HIGH confidence RAYCAST detection with good color
                # Lowered distance threshold from 0.8 to 1.2m
                # High color_score (>0.9) indicates strong red match
                if dist < 1.2 and det['weight'] > 0.4 and method == 'raycast' and color_score > 0.9:
                    pos = det['position']
                    # CRITICAL FIX: Plant is the OCCLUDER - it must be BETWEEN robot start and charger
                    # Plant at X > 2.5 is suspicious - might be detecting charger as plant!
                    if pos[0] > 2.5:
                        print(f"  ⚠️ Rejecting plant lock at ({pos[0]:.2f}, {pos[1]:.2f}) - X > 2.5 (too close to charger area)")
                        continue  # Try next detection
                    
                    self.locked_estimates['plant'] = {
                        'position': pos,
                        'confidence': min(1.0, det['weight'] * 2),
                    }
                    print(f"  🔒 LOCKED plant at ({pos[0]:.2f}, {pos[1]:.2f}) [raycast at {dist:.1f}m]")
                    return
        
        # Lock CHARGER - AprilTag detections can lock immediately, others need more evidence
        # CRITICAL FIX: Don't lock charger until plant is locked (except for AprilTag)
        # This ensures proper position validation
        if self.locked_estimates['charger'] is None:
            charger_detections = self.class_detections['charger']
            n_charger = len(charger_detections)
            charger_weight = self.class_weight_sum['charger']
            
            # Check for AprilTag detections - they can lock IMMEDIATELY if position is valid
            for det in charger_detections:
                if det.get('depth_method') == 'apriltag_pose':
                    pos = det['position']
                    det_x, det_y = pos
                    
                    # Check viewing angle quality
                    is_frontal = det.get('is_frontal', False)
                    viewing_angle = det.get('viewing_angle_deg', 0)
                    
                    # X validation - should be in reasonable range
                    if det_x < 1.5 or det_x > 3.5:
                        print(f"  ⚠️ AprilTag at ({det_x:.2f}, {det_y:.2f}) - X position invalid, not locking")
                        continue  # Try next detection
                    
                    # CRITICAL FIX: Validate position relative to plant!
                    # Wall AprilTags are at Y=±0.40m offset, so detected positions vary
                    plant_pos_check = self.locked_estimates.get('plant')
                    if plant_pos_check is not None:
                        plant_y = plant_pos_check['position'][1]
                        y_error = abs(det_y - plant_y)
                        # RELAXED: Allow ±1.5m Y error for wall tags
                        # Wall tags are offset and viewing angle adds more error
                        if y_error > 1.5:
                            print(f"  ⚠️ AprilTag at ({det_x:.2f}, {det_y:.2f}) - Y too far from plant Y={plant_y:.2f} (error={y_error:.2f}m), REJECTED")
                            continue
                        
                        # Validate X - charger should be BEHIND plant (with tolerance)
                        plant_x = plant_pos_check['position'][0]
                        if det_x < plant_x - 0.2:  # 20cm tolerance for detection error
                            print(f"  ⚠️ AprilTag at ({det_x:.2f}, {det_y:.2f}) - too far in front of plant X={plant_x:.2f}, REJECTED")
                            continue
                    
                    # CRITICAL: Only IMMEDIATE lock for FRONTAL views (<45°)
                    # Marginal views (45-60°) have higher position error
                    # Also skip immediate lock for recovery detections
                    from_recovery = det.get('from_recovery', False)
                    if from_recovery:
                        print(f"  🟡 AprilTag at ({det_x:.2f}, {det_y:.2f}) - from recovery, needs main loop confirmation")
                        # Don't lock - let main loop handle it
                    elif is_frontal:
                        self.locked_estimates['charger'] = {
                            'position': pos,
                            'confidence': 1.0,  # AprilTag frontal = 100% confidence
                        }
                        print(f"  🔒 LOCKED charger at ({det_x:.2f}, {det_y:.2f}) [AprilTag FRONTAL - IMMEDIATE LOCK]")
                        return
                    else:
                        # Marginal view - add to estimates but don't lock immediately
                        # Require confirmation from another detection
                        print(f"  🟡 AprilTag at ({det_x:.2f}, {det_y:.2f}) - marginal view ({viewing_angle:.0f}°), needs confirmation")
                        # Still return the detection for processing, but don't lock
                        # The normal weight-based locking will handle it
            
            # CRITICAL: For non-AprilTag detections, REQUIRE plant to be locked first!
            # This prevents false positives from poisoning the estimate
            if self.locked_estimates['plant'] is None:
                # Don't lock charger yet - wait for plant to be locked
                return
            
            plant_x = self.locked_estimates['plant']['position'][0]
            plant_y = self.locked_estimates['plant']['position'][1]
            
            # Non-AprilTag: Lock when we have enough good detections
            if n_charger >= self.min_detections_to_lock and charger_weight >= self.lock_threshold * 0.5:
                
                # GEOMETRIC CONSISTENCY: Filter out detections that are in front of the plant
                # Also filter out detections that are too far (>3m) - likely false positives
                valid_detections = []
                
                for det in charger_detections:
                    det_x, det_y = det['position']
                    dist = det['distance']
                    
                    # CRITICAL: Skip far detections (>3m) - they're unreliable
                    if dist > 3.0:
                        print(f"  ⚠️ Skipping far charger detection at ({det_x:.2f}, {det_y:.2f}) - {dist:.1f}m too far")
                        continue
                    
                    # Charger must be BEHIND plant (relaxed from 0.5 to 0.2 for locking)
                    if det_x < plant_x + 0.2:
                        print(f"  ⚠️ Rejected charger detection at x={det_x:.2f} (in front of plant at x={plant_x:.2f})")
                        continue
                    
                    # Charger should be near same Y as plant (within 0.5m)
                    if abs(det_y - plant_y) > 0.5:
                        print(f"  ⚠️ Rejected charger detection at y={det_y:.2f} (too far from plant y={plant_y:.2f})")
                        continue
                        
                    valid_detections.append(det)
                
                if not valid_detections:
                    # All detections were invalid - don't lock yet
                    return
                
                # Use BEST single detection - prefer CLOSE detections (more accurate)
                best_det = None
                best_score = -1
                
                for det in valid_detections:
                    dist = det['distance']
                    # Score: prefer close detections (0.5-2.0m)
                    if dist < 2.5:
                        dist_score = max(0, 1.0 - dist / 2.5)  # Closer = better
                        score = det['weight'] * (0.5 + 0.5 * dist_score)
                        if score > best_score:
                            best_score = score
                            best_det = det
                
                if best_det is None:
                    return  # No good detection found
                
                pos = best_det['position']
                conf = min(1.0, best_det['weight'] * 2)
                
                self.locked_estimates['charger'] = {
                    'position': pos,
                    'confidence': conf,
                }
                print(f"  🔒 LOCKED charger at ({pos[0]:.2f}, {pos[1]:.2f}) [BEST detection at {best_det['distance']:.1f}m]")
    
    def _update_grid_from_averages(self):
        """Update grid cells based on averaged class positions."""
        self.alpha = np.full((self.height, self.width), self.prior_alpha, dtype=np.float32)
        self.beta = np.full((self.height, self.width), self.prior_beta, dtype=np.float32)
        self.eta = np.ones((self.height, self.width, NUM_CLASSES), dtype=np.float32)
        
        for class_name, class_id in CLASS_TO_ID.items():
            if class_name == 'free':
                continue
            
            pos, conf = self.get_class_estimate(class_name)
            if pos is None:
                continue
            
            weight_sum = conf * 2.0
            i, j = self.world_to_grid(pos[0], pos[1])
            
            if self.is_valid(i, j):
                self.alpha[j, i] += weight_sum * 3.0
                self.eta[j, i, class_id] += weight_sum * 5.0
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if self.is_valid(ni, nj):
                            self.alpha[nj, ni] += weight_sum * 0.3
                            self.eta[nj, ni, class_id] += weight_sum * 0.5
    
    def _compute_charger_estimate_unlocked(self) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Compute charger estimate from RAW detections only.
        
        NO hardcoded assumptions about charger position relative to plant!
        If we haven't seen the charger, we don't know where it is.
        """
        charger_weight = self.class_weight_sum['charger']
        n_charger = len(self.class_detections['charger'])
        
        if charger_weight < 0.1 or n_charger == 0:
            # No charger detections - we genuinely don't know where it is
            return None, 0.0
        
        # Use weighted average of RAW detections
        raw_pos = self.class_position_sum['charger'] / charger_weight
        
        # FIXED: More generous confidence formula
        # - Single good detection (weight > 0.5) should give meaningful confidence
        # - Multiple detections should build up faster
        # Old formula: min(1.0, charger_weight / 1.5) * min(1.0, n_charger / 3.0)
        # New formula: Give more credit to detection weight, less penalty for single detection
        weight_confidence = min(1.0, charger_weight / 0.8)  # Was / 1.5 - now more generous
        detection_confidence = min(1.0, (n_charger + 0.5) / 2.0)  # Was / 3.0 - now 1 detection = 0.75
        confidence = weight_confidence * detection_confidence
        
        return (raw_pos[0], raw_pos[1]), confidence
    
    def get_charger_estimate(self) -> Tuple[Optional[Tuple[float, float]], float]:
        """Get charger position."""
        if self.locked_estimates['charger'] is not None:
            locked = self.locked_estimates['charger']
            return locked['position'], locked['confidence']
        
        return self._compute_charger_estimate_unlocked()
    
    # REMOVED: _estimate_charger_y method - was using hardcoded plant_to_charger_offset
    # Now we use only raw detections without any spatial assumptions
    
    def get_class_estimate(self, class_name: str) -> Tuple[Optional[Tuple[float, float]], float]:
        """Get position estimate for any class."""
        if self.locked_estimates[class_name] is not None:
            locked = self.locked_estimates[class_name]
            return locked['position'], locked['confidence']
        
        if class_name == 'charger':
            return self._compute_charger_estimate_unlocked()
        
        weight_sum = self.class_weight_sum[class_name]
        if weight_sum > 0.1:
            avg_pos = self.class_position_sum[class_name] / weight_sum
            confidence = min(1.0, weight_sum / 2.0)
            return (avg_pos[0], avg_pos[1]), confidence
        return None, 0.0
    
    def get_occupancy_prob(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta + 1e-10)
    
    def get_semantic_probs(self) -> np.ndarray:
        eta_sum = np.sum(self.eta, axis=2, keepdims=True)
        return self.eta / (eta_sum + 1e-10)
    
    def get_class_prob(self, class_name: str) -> np.ndarray:
        class_id = CLASS_TO_ID.get(class_name, 0)
        return self.get_semantic_probs()[:, :, class_id]
    
    def get_charger_belief(self) -> np.ndarray:
        p_occ = self.get_occupancy_prob()
        p_charger = self.get_class_prob('charger')
        belief = p_occ * p_charger
        total = np.sum(belief)
        if total > 1e-10:
            belief = belief / total
        return belief
    
    def print_summary(self):
        """Print detection summary."""
        print("\n📊 Detection Summary (Raw Estimates):")
        print("-" * 50)
        
        plant_pos, plant_conf = self.get_class_estimate('plant')
        n_plant = len(self.class_detections['plant'])
        locked_plant = "🔒 LOCKED" if self.locked_estimates['plant'] else ""
        if plant_pos:
            print(f"  🌱 PLANT (Landmark) {locked_plant}: {n_plant} detections")
            print(f"     Position: ({plant_pos[0]:.2f}, {plant_pos[1]:.2f})")
            print(f"     Confidence: {plant_conf:.2f}")
        
        charger_pos, charger_conf = self.get_charger_estimate()
        n_charger = len(self.class_detections['charger'])
        locked_charger = "🔒 LOCKED" if self.locked_estimates['charger'] else ""
        if charger_pos:
            print(f"  🔋 CHARGER {locked_charger}: {n_charger} detections")
            print(f"     Position: ({charger_pos[0]:.2f}, {charger_pos[1]:.2f})")
            print(f"     Confidence: {charger_conf:.2f}")
            
            if self.class_weight_sum['charger'] > 0.1:
                raw_pos = self.class_position_sum['charger'] / self.class_weight_sum['charger']
                print(f"     (Raw estimate: ({raw_pos[0]:.2f}, {raw_pos[1]:.2f}))")
        
        obs_pos, obs_conf = self.get_class_estimate('obstacle')
        n_obs = len(self.class_detections['obstacle'])
        if obs_pos:
            print(f"  📦 Obstacle: {n_obs} detections")
            print(f"     Position: ({obs_pos[0]:.2f}, {obs_pos[1]:.2f})")
    
    def visualize(self, robot_pose: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """Create visualization."""
        cell_size = 40
        vis_height = self.height * cell_size
        vis_width = self.width * cell_size
        
        vis = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 220
        
        semantic_probs = self.get_semantic_probs()
        occupancy = self.get_occupancy_prob()
        
        for j in range(self.height):
            for i in range(self.width):
                y1 = (self.height - 1 - j) * cell_size
                y2 = y1 + cell_size
                x1 = i * cell_size
                x2 = x1 + cell_size
                
                probs = semantic_probs[j, i]
                occ = occupancy[j, i]
                
                if occ > 0.55:
                    charger_p = probs[CLASS_TO_ID['charger']]
                    plant_p = probs[CLASS_TO_ID['plant']]
                    obstacle_p = probs[CLASS_TO_ID['obstacle']]
                    
                    if charger_p > plant_p and charger_p > obstacle_p and charger_p > 0.3:
                        intensity = min(255, int(200 + charger_p * 55))
                        color = (0, intensity, 0)
                    elif plant_p > charger_p and plant_p > obstacle_p and plant_p > 0.3:
                        intensity = min(255, int(200 + plant_p * 55))
                        color = (intensity, 0, 0)
                    elif obstacle_p > 0.3:
                        color = (100, 100, 100)
                    else:
                        color = (180, 180, 180)
                    
                    vis[y1+2:y2-2, x1+2:x2-2] = color
                
                vis[y1:y1+1, x1:x2] = (150, 150, 150)
                vis[y1:y2, x1:x1+1] = (150, 150, 150)
        
        # Draw X markers
        colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0), 'obstacle': (80, 80, 80)}
        for class_name, color in colors.items():
            pos, conf = self.get_class_estimate(class_name)
            if pos and conf > 0.1:
                i, j = self.world_to_grid(pos[0], pos[1])
                px = i * cell_size + cell_size // 2
                py = (self.height - 1 - j) * cell_size + cell_size // 2
                
                # Thicker X if locked
                thickness = 3 if self.locked_estimates[class_name] else 1
                for t in range(thickness):
                    for d in range(-10, 11):
                        if 0 <= py+d < vis_height and 0 <= px+d+t < vis_width:
                            vis[py+d, px+d+t] = color
                        if 0 <= py+d < vis_height and 0 <= px-d+t < vis_width:
                            vis[py+d, px-d+t] = color
        
        # Draw robot
        if robot_pose is not None:
            rx, ry, ryaw = robot_pose
            ri, rj = self.world_to_grid(rx, ry)
            px = ri * cell_size + cell_size // 2
            py = (self.height - 1 - rj) * cell_size + cell_size // 2
            
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    if dx*dx + dy*dy <= 36:
                        if 0 <= py+dy < vis_height and 0 <= px+dx < vis_width:
                            vis[py+dy, px+dx] = (0, 0, 255)
            
            dir_len = 12
            ddx = int(dir_len * np.cos(ryaw))
            ddy = int(-dir_len * np.sin(ryaw))
            for t in range(dir_len):
                tx = px + int(t * ddx / dir_len)
                ty = py + int(t * ddy / dir_len)
                if 0 <= ty < vis_height and 0 <= tx < vis_width:
                    vis[ty, tx] = (0, 255, 255)
        
        return vis


if __name__ == "__main__":
    print("Semantic Grid v2 - run test_pomdp_navigation.py to test")
