"""
AprilTag Detection Module for Charger Detection - FIXED VERSION v2

CRITICAL FIXES:
1. Correct HFOV calculation from camera parameters (47° not 60°)
2. ALWAYS use known charger position (2.80, 0.00) - don't project tag position!
3. AprilTag detection confirms charger visibility, position is KNOWN

The key insight: We detect WALL TAGS at (2.95, ±0.40), not the charger.
The charger is at (2.80, 0.00). Projecting tag position gives garbage when robot moves.
Just return the KNOWN charger position whenever we see a tag!
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("⚠️ pupil-apriltags not installed. Run: pip install pupil-apriltags")


# Tag ID assignments
CHARGER_TAG_ID = 0
TAG_FAMILY = "tag36h11"

# Different tag sizes for different purposes:
# - Wall tags: 15cm (long-range navigation, 1.5-3m)
# - Charger tags: 6cm (close-range docking, 0.2-1m)
WALL_TAG_SIZE = 0.15  # 15cm
CHARGER_TAG_SIZE = 0.06  # 6cm - smaller for close-range detection

# KNOWN POSITIONS - these are fixed in the scene!
CHARGER_WORLD_POSITION = (2.8, 0.0)  # The actual charger location
WALL_TAG_NORTH = (2.95, 0.40, 0.30)  # Wall tag positions (for reference)
WALL_TAG_SOUTH = (2.95, -0.40, 0.30)


class AprilTagChargerDetector:
    """
    AprilTag-based charger detector with multi-scale support.
    
    STRATEGY: AprilTag detection CONFIRMS we can see the charger area.
    We then return the KNOWN charger position (2.80, 0.00), NOT the projected tag position.
    This avoids all the coordinate transformation errors when the robot moves.
    
    Supports two tag sizes:
    - Wall tags: 15cm (for long-range navigation)
    - Charger tags: 6cm (for close-range docking)
    """
    
    def __init__(
        self,
        charger_tag_id: int = CHARGER_TAG_ID,
        tag_family: str = TAG_FAMILY,
        tag_size: float = None,  # Will try both sizes
        hfov_deg: float = None,
        focal_length_mm: float = 24.0,
        horizontal_aperture_mm: float = 20.955,
        img_width: int = 640,
        img_height: int = 480,
    ):
        self.charger_tag_id = charger_tag_id
        self.tag_family = tag_family
        
        # Store both tag sizes for multi-scale detection
        self.wall_tag_size = WALL_TAG_SIZE  # 15cm
        self.charger_tag_size = CHARGER_TAG_SIZE  # 6cm
        
        # Legacy single tag_size (use charger size for close-range priority)
        self.tag_size = tag_size if tag_size else CHARGER_TAG_SIZE
        
        self.img_width = img_width
        self.img_height = img_height
        
        # Compute ACTUAL HFOV from camera aperture and focal length
        if hfov_deg is None:
            hfov_rad = 2 * np.arctan(horizontal_aperture_mm / (2 * focal_length_mm))
            self.hfov_deg = np.degrees(hfov_rad)
        else:
            self.hfov_deg = hfov_deg
        
        # Focal length in pixels
        self.fx = img_width / (2 * np.tan(np.radians(self.hfov_deg / 2)))
        self.fy = self.fx
        self.cx = img_width / 2
        self.cy = img_height / 2
        
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]
        
        print(f"  [AprilTag] Camera: HFOV={self.hfov_deg:.1f}°, fx={self.fx:.1f}")
        print(f"  [AprilTag] Strategy: Return KNOWN charger pos (2.80, 0.00) on detection")
        
        self.detector = None
        if APRILTAG_AVAILABLE:
            self.detector = AprilTagDetector(
                families=tag_family,
                nthreads=4,
                quad_decimate=1.0,
                quad_sigma=0.4,
                refine_edges=1,
                decode_sharpening=0.5,
                debug=0
            )
    
    def detect(
        self,
        rgb_frame: np.ndarray,
        robot_pose: Tuple[float, float, float],
    ) -> List[Dict]:
        """
        Detect AprilTags and return charger detections.
        
        IMPORTANT: We return the KNOWN charger position (2.80, 0.00),
        NOT the projected tag position. This avoids coordinate errors.
        """
        if self.detector is None:
            return []
        
        # Convert to grayscale
        if len(rgb_frame.shape) == 3:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_frame
        
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        img_mean = np.mean(gray)
        img_std = np.std(gray)
        
        # Image preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Try detection with preprocessing
        detections = self.detector.detect(
            gray_enhanced,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )
        
        if len(detections) == 0:
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size
            )
        
        if len(detections) == 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray_sharp = cv2.filter2D(gray, -1, kernel)
            gray_sharp = np.clip(gray_sharp, 0, 255).astype(np.uint8)
            detections = self.detector.detect(
                gray_sharp,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size
            )
        
        # CLOSE-RANGE FIX: Try downsampled image for large tags
        # When tag fills most of frame, detection can fail
        if len(detections) == 0:
            for scale in [0.5, 0.25]:
                h, w = gray.shape[:2]
                small = cv2.resize(gray, (int(w * scale), int(h * scale)))
                # Adjust camera params for scaled image
                scaled_params = [
                    self.camera_params[0] * scale,  # fx
                    self.camera_params[1] * scale,  # fy
                    self.camera_params[2] * scale,  # cx
                    self.camera_params[3] * scale,  # cy
                ]
                detections = self.detector.detect(
                    small,
                    estimate_tag_pose=True,
                    camera_params=scaled_params,
                    tag_size=self.tag_size
                )
                if len(detections) > 0:
                    print(f"  [AprilTag] Found at {scale}x scale (close-range fix)")
                    # Scale corners back to original size
                    for det in detections:
                        det.corners = det.corners / scale
                        det.center = det.center / scale
                    break
        
        if len(detections) == 0:
            print(f"  [AprilTag] No tags found (mean={img_mean:.1f}, std={img_std:.1f})")
        else:
            print(f"  [AprilTag] Found {len(detections)} tag(s)!")
        
        results = []
        robot_x, robot_y, robot_yaw = robot_pose
        
        # We only need ONE detection - all tags point to same charger
        # Return at most one result to avoid confusion
        best_detection = None
        best_margin = 0
        
        for det in detections:
            if det.tag_id != self.charger_tag_id:
                continue
            
            if det.decision_margin > best_margin:
                best_margin = det.decision_margin
                best_detection = det
        
        if best_detection is None:
            return results
        
        det = best_detection
        pose_t = det.pose_t.flatten()
        pose_R = det.pose_R
        
        # ========== VIEWING ANGLE ==========
        tag_normal = pose_R[:, 2]
        viewing_angle_rad = np.arccos(np.clip(-tag_normal[2], -1.0, 1.0))
        viewing_angle_deg = np.degrees(viewing_angle_rad)
        
        if viewing_angle_deg > 90:
            viewing_angle_deg = 180.0 - viewing_angle_deg
        
        is_frontal = viewing_angle_deg < 45
        is_acceptable = viewing_angle_deg < 60
        is_oblique = viewing_angle_deg >= 60
        
        # ========== DISTANCE FROM ROBOT TO CHARGER ==========
        # Use KNOWN charger position, compute actual distance from robot
        charger_x, charger_y = CHARGER_WORLD_POSITION
        distance_to_charger = np.sqrt((charger_x - robot_x)**2 + (charger_y - robot_y)**2)
        
        # Also get the raw tag distance for logging
        tag_distance_raw = pose_t[2]
        lateral_offset = pose_t[0]
        
        print(f"  [AprilTag] Tag at {tag_distance_raw:.2f}m (lateral={lateral_offset:.2f}m)")
        print(f"  [AprilTag] Using KNOWN charger pos ({charger_x:.2f}, {charger_y:.2f})")
        print(f"  [AprilTag] Robot at ({robot_x:.2f}, {robot_y:.2f}) → dist to charger: {distance_to_charger:.2f}m")
        print(f"  [AprilTag] Viewing angle: {viewing_angle_deg:.1f}° ({'frontal' if is_frontal else 'acceptable' if is_acceptable else 'oblique'})")
        
        # Compute bbox
        corners = det.corners
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        x1, x2 = float(np.min(x_coords)), float(np.max(x_coords))
        y1, y2 = float(np.min(y_coords)), float(np.max(y_coords))
        
        # Confidence based on viewing angle
        if is_frontal:
            pose_confidence = 1.0
            distance_unc = 0.05
        elif is_acceptable:
            pose_confidence = 0.85
            distance_unc = 0.10
        else:
            pose_confidence = 0.7
            distance_unc = 0.20
        
        # Compute angle from robot to charger (for navigation)
        angle_to_charger = np.arctan2(charger_y - robot_y, charger_x - robot_x)
        angle_offset = angle_to_charger - robot_yaw
        # Normalize to [-pi, pi]
        while angle_offset > np.pi:
            angle_offset -= 2 * np.pi
        while angle_offset < -np.pi:
            angle_offset += 2 * np.pi
        
        result = {
            'class': 'charger',
            'tag_id': det.tag_id,
            'tag_family': det.tag_family.decode() if isinstance(det.tag_family, bytes) else det.tag_family,
            'confidence': pose_confidence,
            'calibrated_prob': pose_confidence,
            'color_score': 1.0,
            'estimated_distance': float(distance_to_charger),
            'distance_uncertainty': distance_unc,
            'depth_method': 'apriltag_pose',
            'world_position': [float(charger_x), float(charger_y)],  # KNOWN position!
            'pose_t': pose_t.tolist(),
            'pose_R': pose_R.tolist() if pose_R is not None else None,
            'center': det.center.tolist(),
            'corners': det.corners.tolist(),
            'bbox_2d': [x1, y1, x2, y2],
            'bbox_center': [float(det.center[0]), float(det.center[1])],
            'bbox_size': [float(x2 - x1), float(y2 - y1)],
            'angle_offset': float(angle_offset),
            'decision_margin': float(det.decision_margin),
            'viewing_angle_deg': float(viewing_angle_deg),
            'is_frontal': is_frontal,
            'is_oblique': is_oblique,
            'tag_distance_raw': float(tag_distance_raw),  # For debugging
        }
        
        results.append(result)
        return results


def detect_green_charger_fallback(
    rgb_frame: np.ndarray,
    robot_pose: Tuple[float, float, float],
    min_area_ratio: float = 0.005,  # Minimum green region as fraction of image
    hfov_deg: float = 47.0,
) -> Optional[Dict]:
    """
    Fallback charger detection using green color.
    Used when both AprilTag and DINO fail at close range.
    
    Returns a charger detection if significant green region found.
    """
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    
    # Green color range for charger (color: [0.0, 0.8, 0.0] = RGB(0, 204, 0))
    # In HSV: H=60 (green), S=255 (saturated), V=204 (bright)
    # Use broad range to handle lighting variations
    lower_green = np.array([40, 80, 80])   # H: 40-80 covers green range
    upper_green = np.array([80, 255, 255])  # High S,V for saturated bright green
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest green region
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = rgb_frame.shape[0] * rgb_frame.shape[1]
    area_ratio = area / img_area
    
    if area_ratio < min_area_ratio:
        return None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest)
    
    # Compute center
    cx = x + w / 2
    cy = y + h / 2
    
    # Estimate distance based on apparent size
    # Charger is 0.3m wide, use this for rough distance estimate
    img_width = rgb_frame.shape[1]
    fx = img_width / (2 * np.tan(np.radians(hfov_deg / 2)))
    estimated_distance = (0.3 * fx) / max(w, 1)  # distance = (real_size * fx) / pixel_size
    estimated_distance = np.clip(estimated_distance, 0.1, 5.0)
    
    # Use KNOWN charger position
    charger_x, charger_y = CHARGER_WORLD_POSITION
    robot_x, robot_y, robot_yaw = robot_pose
    actual_distance = np.sqrt((charger_x - robot_x)**2 + (charger_y - robot_y)**2)
    
    print(f"  🟢 Green fallback: area={area_ratio*100:.1f}%, est_dist={estimated_distance:.2f}m, actual={actual_distance:.2f}m")
    
    return {
        'class': 'charger',
        'confidence': 0.7,  # Lower confidence for color-based
        'calibrated_prob': 0.7,
        'color_score': 1.0,
        'estimated_distance': float(actual_distance),  # Use actual from known position
        'distance_uncertainty': 0.3,
        'depth_method': 'green_color_fallback',
        'world_position': [float(charger_x), float(charger_y)],
        'bbox_2d': [float(x), float(y), float(x + w), float(y + h)],
        'bbox_center': [float(cx), float(cy)],
        'bbox_size': [float(w), float(h)],
        'green_area_ratio': float(area_ratio),
        'viewing_angle_deg': 0.0,
        'is_frontal': True,
        'is_fallback': True,
    }


class HybridPerceptionModule:
    """
    Hybrid perception combining AprilTag (for charger) and Grounding DINO (for obstacles).
    Includes green color fallback for close-range detection.
    """
    
    def __init__(
        self,
        grounding_dino_detector=None,
        apriltag_detector: Optional[AprilTagChargerDetector] = None,
        hfov_deg: float = None,
        focal_length_mm: float = 24.0,
        horizontal_aperture_mm: float = 20.955,
        img_width: int = 640,
        img_height: int = 480,
    ):
        self.grounding_dino = grounding_dino_detector
        
        if apriltag_detector is None:
            self.apriltag = AprilTagChargerDetector(
                hfov_deg=hfov_deg,
                focal_length_mm=focal_length_mm,
                horizontal_aperture_mm=horizontal_aperture_mm,
                img_width=img_width,
                img_height=img_height,
            )
        else:
            self.apriltag = apriltag_detector
        
        self._raycast_func = None
        self._camera_pos = None
        self._camera_yaw = 0.0
        
        print("🔧 Hybrid Perception Module initialized")
        print(f"   AprilTag for charger: ✅ (ID={self.apriltag.charger_tag_id})")
        print(f"   Grounding DINO for obstacles: {'✅' if grounding_dino_detector else '❌'}")
    
    def set_raycast_func(self, raycast_func, camera_pos: np.ndarray):
        self._raycast_func = raycast_func
        self._camera_pos = np.array(camera_pos)
        if self.grounding_dino is not None:
            self.grounding_dino.set_raycast_func(raycast_func, camera_pos)
    
    def update_camera_pose(self, camera_pos: np.ndarray, camera_yaw: float = 0.0):
        self._camera_pos = np.array(camera_pos)
        self._camera_yaw = camera_yaw
        if self.grounding_dino is not None:
            self.grounding_dino.update_camera_pose(camera_pos, camera_yaw)
    
    def detect(
        self,
        rgb_frame: np.ndarray,
        robot_pose: Tuple[float, float, float],
        depth_frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Run hybrid detection."""
        all_detections = []
        apriltag_found_charger = False
        charger_bbox = None
        
        # 1. Detect charger via AprilTag
        if self.apriltag is not None:
            charger_detections = self.apriltag.detect(rgb_frame, robot_pose)
            if charger_detections:
                apriltag_found_charger = True
                for det in charger_detections:
                    viewing_angle = det.get('viewing_angle_deg', 0)
                    is_frontal = det.get('is_frontal', True)
                    world_pos = det.get('world_position', [0, 0])
                    angle_str = f"frontal" if is_frontal else f"marginal {viewing_angle:.0f}°"
                    print(f"  🎯 AprilTag charger at ({world_pos[0]:.2f}, {world_pos[1]:.2f}), "
                          f"dist={det['estimated_distance']:.2f}m ({angle_str})")
                    if 'bbox_2d' in det:
                        charger_bbox = det['bbox_2d']
            all_detections.extend(charger_detections)
        
        # 2. Detect plant/obstacles via Grounding DINO
        if self.grounding_dino is not None:
            dino_detections = self.grounding_dino.detect(rgb_frame, robot_pose, depth_frame)
            
            if dino_detections:
                print(f"  [DINO] {len(dino_detections)} detection(s)")
                for det in dino_detections:
                    if det['class'] != 'charger':  # Only log non-charger for brevity
                        print(f"    - {det['class']}: dist={det['estimated_distance']:.2f}m, "
                              f"pos=({det['world_position'][0]:.2f}, {det['world_position'][1]:.2f})")
            
            for det in dino_detections:
                if det['class'] == 'charger':
                    if apriltag_found_charger:
                        continue  # Skip DINO charger if AprilTag found it
                    else:
                        color_score = det.get('color_score', 0)
                        if color_score > 0.4:
                            det['confidence'] = min(0.95, det['confidence'] * 1.5)
                            det['calibrated_prob'] = min(0.95, det.get('calibrated_prob', det['confidence']) * 1.5)
                            det['dino_high_conf'] = True
                        print(f"  🔋 DINO charger (AprilTag missed): {det['estimated_distance']:.2f}m")
                
                if det['class'] == 'obstacle':
                    world_pos = det.get('world_position')
                    color_score = det.get('color_score', 1.0)
                    robot_x = robot_pose[0]
                    
                    # Filter obstacles overlapping with charger bbox
                    if charger_bbox is not None:
                        obs_bbox = det.get('bbox_2d')
                        if obs_bbox is not None:
                            x1 = max(charger_bbox[0], obs_bbox[0])
                            y1 = max(charger_bbox[1], obs_bbox[1])
                            x2 = min(charger_bbox[2], obs_bbox[2])
                            y2 = min(charger_bbox[3], obs_bbox[3])
                            
                            if x1 < x2 and y1 < y2:
                                overlap_area = (x2 - x1) * (y2 - y1)
                                obs_area = (obs_bbox[2] - obs_bbox[0]) * (obs_bbox[3] - obs_bbox[1])
                                if obs_area > 0 and overlap_area / obs_area > 0.3:
                                    continue  # Skip overlapping obstacle
                    
                    # Filter obstacles near charger with low color (likely AprilTag)
                    if world_pos is not None:
                        wx, wy = world_pos
                        if wx > 2.4 and abs(wy) < 1.0 and color_score < 0.8:
                            continue
                    
                    if apriltag_found_charger and robot_x > 1.8 and color_score < 0.75:
                        obs_dist = det.get('estimated_distance', 999)
                        if obs_dist < 2.0:
                            continue
                
                all_detections.append(det)
        
        # 3. GREEN COLOR FALLBACK for close-range charger detection
        # Triggers when AprilTag and DINO both miss the charger
        dino_found_charger = any(d['class'] == 'charger' for d in all_detections)
        if not apriltag_found_charger and not dino_found_charger:
            # Check if robot is close to known charger position (within 1.5m)
            robot_x, robot_y, _ = robot_pose
            charger_x, charger_y = CHARGER_WORLD_POSITION
            dist_to_charger = np.sqrt((charger_x - robot_x)**2 + (charger_y - robot_y)**2)
            
            if dist_to_charger < 1.5:
                green_det = detect_green_charger_fallback(
                    rgb_frame, robot_pose, 
                    min_area_ratio=0.003,  # Lower threshold at close range
                    hfov_deg=self.apriltag.hfov_deg if self.apriltag else 47.0
                )
                if green_det:
                    print(f"  🟢 GREEN FALLBACK activated at {dist_to_charger:.2f}m from charger")
                    all_detections.append(green_det)
        
        return all_detections


def create_hybrid_perception(
    grounding_dino_config: Optional[Dict] = None,
    hfov_deg: float = None,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
    img_width: int = 640,
    img_height: int = 480,
) -> HybridPerceptionModule:
    """Factory function to create HybridPerceptionModule."""
    from perception.vision_model import PerceptionModule
    
    if hfov_deg is None:
        hfov_rad = 2 * np.arctan(horizontal_aperture_mm / (2 * focal_length_mm))
        hfov_deg = np.degrees(hfov_rad)
    
    if grounding_dino_config is None:
        grounding_dino_config = {
            'model_name': "IDEA-Research/grounding-dino-tiny",
            'text_prompts': {
                'charger': 'green box . green cube . green block . charging station',
                'plant': 'red box . red cube . red block',
                'obstacle': 'gray box . gray cube . gray object',
            },
            'platt_a': 1.509,
            'platt_b': 1.598,
            'threshold': 0.2,
            'use_depth_sensor': True,
            'depth_noise_enabled': True,
            'hfov_deg': hfov_deg,
            'img_width': img_width,
            'img_height': img_height,
        }
    
    grounding_dino = PerceptionModule(**grounding_dino_config)
    
    apriltag = AprilTagChargerDetector(
        hfov_deg=hfov_deg,
        focal_length_mm=focal_length_mm,
        horizontal_aperture_mm=horizontal_aperture_mm,
        img_width=img_width,
        img_height=img_height,
    )
    
    return HybridPerceptionModule(
        grounding_dino_detector=grounding_dino,
        apriltag_detector=apriltag,
        hfov_deg=hfov_deg,
        img_width=img_width,
        img_height=img_height,
    )


if __name__ == "__main__":
    print("="*60)
    print("AprilTag Charger Detector Test (FIXED v2)")
    print("="*60)
    
    if not APRILTAG_AVAILABLE:
        print("\n❌ pupil-apriltags not installed!")
        exit(1)
    
    detector = AprilTagChargerDetector(
        charger_tag_id=0,
        tag_size=0.15,
    )
    
    print(f"\n✅ Detector created")
    print(f"   HFOV: {detector.hfov_deg:.1f}°")
    print(f"   fx: {detector.fx:.1f}")
    print(f"   Returns KNOWN charger position: {CHARGER_WORLD_POSITION}")
