"""
Perception Module: Grounding DINO with Depth Sensor Integration.

Detects:
- Charger (GREEN box) - goal
- Plant (RED box) - obstacle  
- Obstacles (GRAY boxes) - distractors

Depth estimation:
- PRIMARY: Raycast-based depth with realistic sensor noise (accurate)
- FALLBACK: Bbox-based estimation (inaccurate, ~40% error)

Calibration:
- Uses Platt scaling for confidence calibration
- Parameters from calibration/run_ece_calibration.py
- ECE improved from 0.34 to 0.07 (79% improvement)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Callable
from PIL import Image
import warnings

# Suppress the FutureWarning from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Known object sizes (width in meters) for depth estimation
KNOWN_OBJECT_SIZES = {
    'charger': {'width': 0.3, 'half_width': 0.15},
    'plant': {'width': 0.4, 'half_width': 0.2},
    'obstacle': {'width': 0.35, 'half_width': 0.175},
    'unknown': {'width': 0.4, 'half_width': 0.2},
}

# Default Platt scaling parameters (from calibration)
# calibrated_prob = sigmoid(a * logit(raw_prob) + b)
DEFAULT_PLATT_A = 1.509
DEFAULT_PLATT_B = 1.598


class RealisticDepthSensor:
    """
    Simulates depth sensor with CHALLENGING noise characteristics.
    Uses raycast for ground truth, adds significant noise.
    
    Default noise (~8-12% error at typical distances):
    - 5cm base noise + 4% of distance
    - 5% random dropouts
    - Simulates dusty lens, poor lighting, degraded sensor
    """
    
    def __init__(
        self,
        base_noise_std: float = 0.05,
        distance_noise_factor: float = 0.04,
        quantization_step: float = 0.005,
        random_invalid_prob: float = 0.05,
        min_valid_range: float = 0.4,
        max_valid_range: float = 10.0,
        hfov_deg: float = 60.0,
        img_width: int = 640,
        img_height: int = 480,
    ):
        self.base_noise_std = base_noise_std
        self.distance_noise_factor = distance_noise_factor
        self.quantization_step = quantization_step
        self.random_invalid_prob = random_invalid_prob
        self.min_valid_range = min_valid_range
        self.max_valid_range = max_valid_range
        self.hfov_deg = hfov_deg
        self.img_width = img_width
        self.img_height = img_height
        
        import math
        self.focal_length_pixels = img_width / (2 * math.tan(math.radians(hfov_deg / 2)))
        
    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Convert pixel coordinates to ray direction."""
        import math
        
        hfov = math.radians(self.hfov_deg)
        vfov = hfov * (self.img_height / self.img_width)
        
        nx = (u - self.img_width / 2) / (self.img_width / 2)
        ny = (self.img_height / 2 - v) / (self.img_height / 2)
        
        angle_h = nx * (hfov / 2)
        angle_v = ny * (vfov / 2)
        
        dx = math.cos(angle_v) * math.cos(angle_h)
        dy = -math.cos(angle_v) * math.sin(angle_h)
        dz = math.sin(angle_v)
        
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        return np.array([dx/length, dy/length, dz/length])
    
    def add_realistic_noise(self, depth: float) -> float:
        """Add realistic noise to a depth measurement."""
        if depth <= 0:
            return 0.0
        
        noise_std = self.base_noise_std + self.distance_noise_factor * depth
        noisy_depth = depth + np.random.normal(0, noise_std)
        noisy_depth = round(noisy_depth / self.quantization_step) * self.quantization_step
        
        return max(0, noisy_depth)
    
    def get_depth_at_pixel(
        self, 
        cam_pos: np.ndarray, 
        u: float, 
        v: float,
        raycast_func: Callable,
        add_noise: bool = True
    ) -> Tuple[float, bool, Optional[str]]:
        """Get depth at a pixel location using raycast."""
        ray_dir = self.pixel_to_ray(u, v)
        distance, hit_body = raycast_func(cam_pos, ray_dir)
        
        if distance is None:
            return 0.0, False, None
        
        if distance < self.min_valid_range or distance > self.max_valid_range:
            return 0.0, False, hit_body
        
        if add_noise and np.random.random() < self.random_invalid_prob:
            return 0.0, False, hit_body
        
        if add_noise:
            distance = self.add_realistic_noise(distance)
        
        return distance, True, hit_body


class PerceptionModule:
    """
    Grounding DINO perception with Platt-calibrated confidences.
    
    Calibration uses Platt scaling: P(correct) = sigmoid(a * logit(conf) + b)
    This shifts underconfident predictions up to match actual accuracy.
    
    Calibration results (from run_ece_calibration.py):
    - Raw accuracy: 81.8%
    - Raw mean confidence: 49.7% (underconfident!)
    - ECE before: 0.34
    - ECE after Platt: 0.07 (79% improvement)
    """
    
    def __init__(
        self, 
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        text_prompts: Optional[Dict[str, str]] = None, 
        # Calibration parameters
        platt_a: float = DEFAULT_PLATT_A,  # Platt scaling slope
        platt_b: float = DEFAULT_PLATT_B,  # Platt scaling intercept
        # Legacy parameter (ignored, kept for compatibility)
        temperature: float = 2.0,
        device: str = 'cuda',
        threshold: float = 0.3,
        # Depth sensor parameters
        use_depth_sensor: bool = True,
        depth_noise_enabled: bool = True,
        hfov_deg: float = 60.0,
        img_width: int = 640,
        img_height: int = 480,
    ):
        """
        Initialize perception module with Platt-calibrated confidences.
        
        Args:
            platt_a: Platt scaling slope (default from calibration)
            platt_b: Platt scaling intercept (default from calibration)
            temperature: DEPRECATED - kept for compatibility, use platt_a/platt_b
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.threshold = threshold
        self.use_depth_sensor = use_depth_sensor
        self.depth_noise_enabled = depth_noise_enabled
        
        # Default prompts
        # CRITICAL: Do NOT include "white" in obstacle prompt!
        # AprilTag on charger has white squares that would match "white box"
        if text_prompts is None:
            text_prompts = {
                'charger': 'green box . green cube . green block . green charging station',
                'plant': 'red box . red cube . red block . red pot',
                'obstacle': 'gray box . gray cube . gray block',  # NO WHITE - AprilTag has white squares!
            }
        self.text_prompts = text_prompts
        
        self.color_requirements = {
            'charger': 'green',
            'plant': 'red',
            'obstacle': 'gray',
        }
        
        # Camera parameters
        self.image_width = img_width
        self.image_height = img_height
        self.fov_horizontal = hfov_deg
        self.focal_length_pixels = self.image_width / (2 * np.tan(np.radians(self.fov_horizontal / 2)))
        
        # Initialize depth sensor
        self.depth_sensor = RealisticDepthSensor(
            hfov_deg=hfov_deg,
            img_width=img_width,
            img_height=img_height,
        )
        
        self._raycast_func = None
        self._camera_pos = None
        
        print(f"Perception Module Initializing...")
        print(f"  Device: {self.device}")
        print(f"  Calibration: Platt scaling (a={platt_a:.3f}, b={platt_b:.3f})")
        print(f"  Threshold: {threshold}")
        print(f"  Classes: {list(text_prompts.keys())}")
        print(f"  Depth sensor: {'ENABLED' if use_depth_sensor else 'DISABLED'}")
        print(f"  Depth noise: {'ENABLED' if depth_noise_enabled else 'DISABLED'}")
        
        self._load_model(model_name)
    
    def set_raycast_func(self, raycast_func: Callable, camera_pos: np.ndarray):
        """Set raycast function for depth measurement."""
        self._raycast_func = raycast_func
        self._camera_pos = np.array(camera_pos)
        self._camera_yaw = 0.0
        print(f"  ✓ Raycast depth enabled at camera pos {camera_pos}")
    
    def update_camera_pose(self, camera_pos: np.ndarray, camera_yaw: float = 0.0):
        """Update camera position and orientation."""
        self._camera_pos = np.array(camera_pos)
        self._camera_yaw = camera_yaw
    
    def _load_model(self, model_name: str):
        """Load Grounding DINO model."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            print(f"  Loading model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            
            print("  ✓ Model loaded successfully")
            self.model_loaded = True
            
        except Exception as e:
            print(f"  ⚠ Could not load model: {e}")
            print("  → Running in MOCK mode")
            self.model_loaded = False
    
    def _check_color_in_bbox(self, image: np.ndarray, bbox: List[float], target_color: str) -> float:
        """Check if bbox region contains the target color."""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        region = image[y1:y2, x1:x2]
        r = region[:, :, 0].astype(float)
        g = region[:, :, 1].astype(float)
        b = region[:, :, 2].astype(float)
        
        if target_color == 'green':
            # Green detection for charger
            # Note: Charger may have AprilTag (black/white) which dilutes green pixels
            green_dominance = g - np.maximum(r, b)
            # Relaxed requirements:
            # 1. Green channel > 80 (bright enough) - reduced from 100
            # 2. Green dominance > 30 (clearly green) - reduced from 40
            # 3. Not too white/gray (saturation check)
            max_channel = np.maximum(r, np.maximum(g, b))
            min_channel = np.minimum(r, np.minimum(g, b))
            saturation = (max_channel - min_channel) / (max_channel + 1e-6)
            is_green = (
                (g > 80) & 
                (green_dominance > 30) & 
                (saturation > 0.3)  # Must have some color (not gray/white)
            )
            return np.mean(is_green)
            
        elif target_color == 'red':
            red_dominance = r - np.maximum(g, b)
            is_red = (r > 80) & (red_dominance > 20)
            return np.mean(is_red)
            
        elif target_color == 'gray':
            max_channel = np.maximum(r, np.maximum(g, b))
            min_channel = np.minimum(r, np.minimum(g, b))
            saturation = (max_channel - min_channel) / (max_channel + 1e-6)
            is_gray = (saturation < 0.3) & (max_channel > 50) & (max_channel < 250)
            return np.mean(is_gray)
        
        return 0.5
    
    def detect(
        self, 
        rgb_frame: np.ndarray,
        robot_pose: Tuple[float, float, float],
        depth_frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Run detection on RGB frame with color verification and depth estimation.
        
        Returns list of detections with calibrated confidence scores.
        """
        if not self.model_loaded:
            return self._mock_detect(rgb_frame, robot_pose)
        
        all_detections = []
        for class_name, prompt in self.text_prompts.items():
            class_dets = self._detect_class(rgb_frame, robot_pose, class_name, prompt)
            all_detections.extend(class_dets)
        
        detections = self._remove_duplicates(all_detections)
        return detections
    
    def _remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate detections based on IoU."""
        if len(detections) <= 1:
            return detections
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        for det in detections:
            is_duplicate = False
            for kept in keep:
                iou = self._compute_iou(det['bbox_2d'], kept['bbox_2d'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(det)
        return keep
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute intersection over union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _detect_class(
        self,
        rgb_frame: np.ndarray,
        robot_pose: Tuple[float, float, float],
        class_name: str,
        prompt: str,
    ) -> List[Dict]:
        """Detect a single class with color verification."""
        detections = []
        
        try:
            if isinstance(rgb_frame, np.ndarray):
                image = Image.fromarray(rgb_frame)
            else:
                image = rgb_frame
            
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                threshold=self.threshold,
                target_sizes=[(rgb_frame.shape[0], rgb_frame.shape[1])]
            )[0]
            
            target_color = self.color_requirements.get(class_name)
            
            for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
                if isinstance(box, torch.Tensor):
                    box_np = box.cpu().numpy()
                else:
                    box_np = np.array(box)
                
                x1, y1, x2, y2 = box_np
                
                if target_color:
                    color_score = self._check_color_in_bbox(rgb_frame, [x1, y1, x2, y2], target_color)
                    
                    if class_name in ['charger', 'plant']:
                        # RELAXED: Charger threshold lowered to 0.15 because AprilTag 
                        # covers most of the visible face (black/white pattern dilutes green)
                        # Plant threshold kept at 0.5 since it's fully red
                        min_color_threshold = 0.15 if class_name == 'charger' else 0.5
                        if color_score < min_color_threshold:
                            continue
                    elif class_name == 'obstacle':
                        green_score = self._check_color_in_bbox(rgb_frame, [x1, y1, x2, y2], 'green')
                        red_score = self._check_color_in_bbox(rgb_frame, [x1, y1, x2, y2], 'red')
                        if green_score > 0.15 or red_score > 0.15:
                            continue
                        if color_score < 0.3:
                            continue
                else:
                    color_score = 1.0
                
                det = self._process_detection(box, score, class_name, robot_pose, color_score)
                if det is not None:
                    detections.append(det)
                    
        except Exception as e:
            print(f"⚠ Detection error for {class_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def _get_depth_for_detection(
        self, 
        center_x: float, 
        center_y: float, 
        class_name: str,
        bbox_width: float,
    ) -> Tuple[float, float, str]:
        """Get depth for a detection using raycast or bbox fallback."""
        if (self.use_depth_sensor and 
            self._raycast_func is not None and 
            self._camera_pos is not None):
            
            ray_local = self.depth_sensor.pixel_to_ray(center_x, center_y)
            yaw = getattr(self, '_camera_yaw', 0.0)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            ray_world = np.array([
                ray_local[0] * cos_yaw - ray_local[1] * sin_yaw,
                ray_local[0] * sin_yaw + ray_local[1] * cos_yaw,
                ray_local[2]
            ])
            
            distance, hit_body = self._raycast_func(self._camera_pos, ray_world)
            
            if distance is not None:
                if self.depth_noise_enabled:
                    distance = self.depth_sensor.add_realistic_noise(distance)
                    if np.random.random() < self.depth_sensor.random_invalid_prob:
                        distance = None
                        
            if distance is not None and distance > 0:
                object_info = KNOWN_OBJECT_SIZES.get(class_name, KNOWN_OBJECT_SIZES['unknown'])
                half_width = object_info['half_width']
                center_distance = distance + half_width
                
                if self.depth_noise_enabled:
                    uncertainty = 0.05 + 0.04 * distance
                else:
                    uncertainty = 0.01
                
                return center_distance, uncertainty, 'raycast'
        
        # Fallback: bbox-based depth
        object_info = KNOWN_OBJECT_SIZES.get(class_name, KNOWN_OBJECT_SIZES['unknown'])
        known_width = object_info['width']
        
        if bbox_width > 10:
            estimated_distance = (known_width * self.focal_length_pixels) / bbox_width
        else:
            estimated_distance = 10.0
        
        uncertainty = 0.2 + 0.1 * estimated_distance**2
        return estimated_distance, uncertainty, 'bbox'
    
    def _process_detection(
        self, 
        box: torch.Tensor, 
        score: torch.Tensor, 
        class_name: str,
        robot_pose: Tuple[float, float, float],
        color_score: float = 1.0,
    ) -> Optional[Dict]:
        """Process a single detection with depth estimation and Platt calibration."""
        if isinstance(box, torch.Tensor):
            box_np = box.cpu().numpy()
        else:
            box_np = np.array(box)
            
        x1, y1, x2, y2 = box_np
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        estimated_distance, distance_uncertainty, depth_method = self._get_depth_for_detection(
            center_x, center_y, class_name, bbox_width
        )
        
        if isinstance(score, torch.Tensor):
            raw_score = score.item()
        else:
            raw_score = float(score)
        
        # Combine detector confidence with color score
        combined_score = raw_score * (0.5 + 0.5 * color_score)
        
        # Apply Platt scaling calibration
        calibrated_prob = self._calibrate_probability_platt(combined_score)
        
        # World position
        robot_x, robot_y, robot_yaw = robot_pose
        angle_offset = np.arctan2(
            (self.image_width / 2 - center_x),
            self.focal_length_pixels
        )
        
        world_angle = robot_yaw + angle_offset
        world_x = robot_x + estimated_distance * np.cos(world_angle)
        world_y = robot_y + estimated_distance * np.sin(world_angle)
        
        return {
            'class': class_name,
            'bbox_2d': [float(x1), float(y1), float(x2), float(y2)],
            'bbox_center': [float(center_x), float(center_y)],
            'bbox_size': [float(bbox_width), float(bbox_height)],
            'confidence': float(raw_score),
            'color_score': float(color_score),
            'calibrated_prob': float(calibrated_prob),
            'estimated_distance': float(estimated_distance),
            'distance_uncertainty': float(distance_uncertainty),
            'depth_method': depth_method,
            'world_position': [float(world_x), float(world_y)],
            'angle_offset': float(angle_offset),
        }
    
    def _calibrate_probability_platt(self, raw_prob: float) -> float:
        """
        Apply Platt scaling calibration.
        
        Formula: calibrated = sigmoid(a * logit(raw) + b)
        
        This can both scale AND shift confidences, unlike temperature scaling.
        Our detector is underconfident (says 50%, correct 82%), so b > 0 shifts up.
        """
        epsilon = 1e-7
        raw_prob = np.clip(raw_prob, epsilon, 1 - epsilon)
        
        # Convert to logit
        logit = np.log(raw_prob / (1 - raw_prob))
        
        # Apply Platt scaling: a * logit + b
        scaled_logit = self.platt_a * logit + self.platt_b
        
        # Convert back to probability
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(calibrated)
    
    def _mock_detect(
        self, 
        rgb_frame: np.ndarray, 
        robot_pose: Tuple[float, float, float]
    ) -> List[Dict]:
        """Mock detection for testing without model."""
        robot_x, robot_y, robot_yaw = robot_pose
        detections = []
        
        mock_objects = [
            {'class': 'charger', 'world_pos': [2.8, 0.0], 'size': 0.3},
            {'class': 'plant', 'world_pos': [2.0, 0.0], 'size': 0.4},
            {'class': 'obstacle', 'world_pos': [2.0, 1.5], 'size': 0.35},
            {'class': 'obstacle', 'world_pos': [2.0, -1.5], 'size': 0.35},
        ]
        
        for obj in mock_objects:
            obj_x, obj_y = obj['world_pos']
            rel_x = obj_x - robot_x
            rel_y = obj_y - robot_y
            
            cos_yaw = np.cos(-robot_yaw)
            sin_yaw = np.sin(-robot_yaw)
            local_x = rel_x * cos_yaw - rel_y * sin_yaw
            local_y = rel_x * sin_yaw + rel_y * cos_yaw
            
            if local_x > 0.5:
                angle = np.arctan2(local_y, local_x)
                if abs(angle) < np.radians(self.fov_horizontal / 2):
                    distance = np.sqrt(local_x**2 + local_y**2)
                    center_x = self.image_width / 2 + local_y * self.focal_length_pixels / local_x
                    center_y = self.image_height / 2
                    
                    bbox_width = (obj['size'] * self.focal_length_pixels) / distance
                    bbox_height = bbox_width * 1.5
                    
                    est_distance, uncertainty, depth_method = self._get_depth_for_detection(
                        center_x, center_y, obj['class'], bbox_width
                    )
                    
                    if depth_method == 'bbox':
                        est_distance = distance
                        uncertainty = 0.05
                        depth_method = 'mock_true'
                    
                    base_conf = 0.9 * np.exp(-distance / 5.0)
                    confidence = np.clip(base_conf + np.random.normal(0, 0.05), 0.1, 0.95)
                    
                    detections.append({
                        'class': obj['class'],
                        'bbox_2d': [
                            center_x - bbox_width/2,
                            center_y - bbox_height/2,
                            center_x + bbox_width/2,
                            center_y + bbox_height/2,
                        ],
                        'bbox_center': [center_x, center_y],
                        'bbox_size': [bbox_width, bbox_height],
                        'confidence': confidence,
                        'color_score': 1.0,
                        'calibrated_prob': self._calibrate_probability_platt(confidence),
                        'estimated_distance': est_distance,
                        'distance_uncertainty': uncertainty,
                        'depth_method': depth_method,
                        'world_position': [obj_x, obj_y],
                        'angle_offset': angle,
                    })
        
        return detections


# Helper function to create raycast wrapper for Isaac Sim
def create_isaac_raycast_func(physx_query_interface, max_distance: float = 20.0):
    """
    Create a raycast function compatible with PerceptionModule.
    
    Usage:
        from omni.physx import get_physx_scene_query_interface
        physx_query = get_physx_scene_query_interface()
        raycast_func = create_isaac_raycast_func(physx_query)
        perception.set_raycast_func(raycast_func, camera_pos)
    """
    def raycast(origin: np.ndarray, direction: np.ndarray):
        origin_tuple = tuple(float(x) for x in origin)
        dir_tuple = tuple(float(x) for x in direction)
        
        hit = physx_query_interface.raycast_closest(origin_tuple, dir_tuple, max_distance)
        
        if hit["hit"]:
            return hit["distance"], hit.get("rigidBody", "unknown")
        return None, None
    
    return raycast


# Utility function to load calibration from file
def load_calibration_params(config_path: str = None) -> Tuple[float, float]:
    """
    Load Platt scaling parameters from calibration config.
    
    Returns (platt_a, platt_b) tuple.
    """
    import os
    import yaml
    
    if config_path is None:
        # Default path relative to this file
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'calibration.yaml'
        )
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('platt_a', DEFAULT_PLATT_A), config.get('platt_b', DEFAULT_PLATT_B)
    
    return DEFAULT_PLATT_A, DEFAULT_PLATT_B


if __name__ == "__main__":
    print("="*60)
    print("Testing Perception Module with Platt Scaling")
    print("="*60)
    
    # Test Platt scaling
    print("\n[Test 1] Platt scaling effect:")
    perception = PerceptionModule(
        model_name="IDEA-Research/grounding-dino-tiny",
        platt_a=DEFAULT_PLATT_A,
        platt_b=DEFAULT_PLATT_B,
        device='cpu',
    )
    
    print(f"  Platt params: a={DEFAULT_PLATT_A}, b={DEFAULT_PLATT_B}")
    print(f"\n  Raw → Calibrated:")
    for raw in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        cal = perception._calibrate_probability_platt(raw)
        print(f"    {raw:.1%} → {cal:.1%}")
    
    # Test detection
    print("\n[Test 2] Mock detection:")
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    robot_pose = (0.0, 0.0, 0.0)
    
    detections = perception.detect(rgb, robot_pose)
    
    print(f"\nDetected {len(detections)} objects:")
    for det in detections:
        print(f"  {det['class']}:")
        print(f"    Raw conf: {det['confidence']:.1%}")
        print(f"    Calibrated: {det['calibrated_prob']:.1%}")
        print(f"    Distance: {det['estimated_distance']:.2f}m")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)
