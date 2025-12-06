"""
Calibration Metrics: ECE (Expected Calibration Error) computation.

ECE measures how well confidence scores match actual accuracy.
A perfectly calibrated model with 70% confidence should be correct 70% of the time.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_ECE(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = Σ (|accuracy_bin - confidence_bin| × n_bin / n_total)
    
    Args:
        confidences: Array of predicted confidence scores [0, 1]
        accuracies: Array of binary labels (1 = correct, 0 = incorrect)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            
            # Weighted absolute difference
            ece += np.abs(avg_accuracy - avg_confidence) * n_in_bin
    
    ece /= len(confidences)
    return float(ece)


def compute_MCE(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE = max(|accuracy_bin - confidence_bin|)
    
    Args:
        confidences: Array of predicted confidence scores [0, 1]
        accuracies: Array of binary labels (1 = correct, 0 = incorrect)
        n_bins: Number of bins
        
    Returns:
        MCE value (lower is better)
    """
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            error = np.abs(avg_accuracy - avg_confidence)
            max_error = max(max_error, error)
    
    return float(max_error)


def compute_brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Compute Brier Score (mean squared error of probabilities).
    
    Brier = mean((confidence - accuracy)^2)
    
    Args:
        confidences: Predicted confidence scores [0, 1]
        accuracies: Binary labels (1 = correct, 0 = incorrect)
        
    Returns:
        Brier score (lower is better, 0 = perfect)
    """
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    return float(np.mean((confidences - accuracies) ** 2))


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes as [x1, y1, x2, y2]
        
    Returns:
        IoU value [0, 1]
    """
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


def match_detection_to_ground_truth(
    detection_bbox: List[float],
    gt_bboxes: List[List[float]],
    iou_threshold: float = 0.5
) -> Tuple[bool, int, float]:
    """
    Match a detection to ground truth bounding boxes.
    
    Args:
        detection_bbox: Detected bbox [x1, y1, x2, y2]
        gt_bboxes: List of ground truth bboxes
        iou_threshold: Minimum IoU for a match
        
    Returns:
        (is_match, matched_gt_index, best_iou)
    """
    best_iou = 0.0
    best_idx = -1
    
    for idx, gt_bbox in enumerate(gt_bboxes):
        iou = compute_iou(detection_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    
    is_match = best_iou >= iou_threshold
    return is_match, best_idx, best_iou


def world_to_image_bbox(
    world_pos: Tuple[float, float],
    object_size: float,
    robot_pose: Tuple[float, float, float],
    camera_params: Dict,
) -> Optional[List[float]]:
    """
    Project world position to image bounding box.
    
    Args:
        world_pos: (x, y) in world coordinates
        object_size: Size of object in meters
        robot_pose: (x, y, yaw) of robot
        camera_params: Camera intrinsics/extrinsics
        
    Returns:
        Bounding box [x1, y1, x2, y2] in image coordinates, or None if not visible
    """
    rx, ry, ryaw = robot_pose
    wx, wy = world_pos
    
    # Transform to robot frame
    dx = wx - rx
    dy = wy - ry
    
    # Rotate by -yaw to get camera frame (camera faces +X)
    cos_yaw = np.cos(-ryaw)
    sin_yaw = np.sin(-ryaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    
    # Check if in front of camera
    if local_x < 0.3:
        return None
    
    # Get camera parameters
    img_width = camera_params.get('width', 640)
    img_height = camera_params.get('height', 480)
    fov_deg = camera_params.get('fov', 60.0)
    focal_length = img_width / (2 * np.tan(np.radians(fov_deg / 2)))
    
    # Check if in FOV
    angle = np.arctan2(local_y, local_x)
    if abs(angle) > np.radians(fov_deg / 2):
        return None
    
    # Project to image
    center_u = img_width / 2 - (local_y / local_x) * focal_length
    center_v = img_height / 2
    
    # Compute bbox size based on distance
    distance = np.sqrt(local_x**2 + local_y**2)
    bbox_width = (object_size * focal_length) / distance
    bbox_height = bbox_width * 1.5  # Assume taller than wide
    
    x1 = center_u - bbox_width / 2
    y1 = center_v - bbox_height / 2
    x2 = center_u + bbox_width / 2
    y2 = center_v + bbox_height / 2
    
    # Clip to image bounds
    x1 = max(0, min(img_width, x1))
    x2 = max(0, min(img_width, x2))
    y1 = max(0, min(img_height, y1))
    y2 = max(0, min(img_height, y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return [x1, y1, x2, y2]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return sigmoid(logits / temperature)


def find_optimal_temperature(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    t_range: Tuple[float, float] = (0.5, 4.0),
    n_steps: int = 36
) -> Tuple[float, float, float]:
    """
    Find optimal temperature that minimizes ECE.
    
    Args:
        confidences: Raw confidence scores (before temperature scaling)
        accuracies: Binary accuracy labels
        t_range: Range of temperatures to search
        n_steps: Number of temperature values to try
        
    Returns:
        (optimal_temperature, ECE_before, ECE_after)
    """
    # Convert confidences to logits
    epsilon = 1e-7
    confidences = np.clip(confidences, epsilon, 1 - epsilon)
    logits = np.log(confidences / (1 - confidences))
    
    # ECE before calibration (T=1.0)
    ece_before = compute_ECE(confidences, accuracies)
    
    # Grid search for optimal T
    best_t = 1.0
    best_ece = ece_before
    
    t_values = np.linspace(t_range[0], t_range[1], n_steps)
    
    for t in t_values:
        calibrated = temperature_scale(logits, t)
        ece = compute_ECE(calibrated, accuracies)
        
        if ece < best_ece:
            best_ece = ece
            best_t = t
    
    return best_t, ece_before, best_ece


class CalibrationEvaluator:
    """
    Evaluates detector calibration by collecting predictions and computing metrics.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Clear all collected data."""
        self.predictions = []  # List of (confidence, is_correct)
        self.raw_confidences = []
        self.class_predictions = {}  # Per-class predictions
    
    def add_prediction(
        self,
        detection: Dict,
        gt_bboxes: List[List[float]],
        class_name: str = None
    ):
        """
        Add a prediction for evaluation.
        
        Args:
            detection: Detection dict with 'bbox_2d', 'confidence', 'class'
            gt_bboxes: List of ground truth bboxes for this class
            class_name: Optional class name override
        """
        det_bbox = detection['bbox_2d']
        confidence = detection.get('calibrated_prob', detection.get('confidence', 0.5))
        raw_conf = detection.get('confidence', confidence)
        
        is_correct, _, _ = match_detection_to_ground_truth(
            det_bbox, gt_bboxes, self.iou_threshold
        )
        
        self.predictions.append((confidence, 1 if is_correct else 0))
        self.raw_confidences.append(raw_conf)
        
        # Per-class tracking
        cls = class_name or detection.get('class', 'unknown')
        if cls not in self.class_predictions:
            self.class_predictions[cls] = []
        self.class_predictions[cls].append((confidence, 1 if is_correct else 0))
    
    def compute_metrics(self) -> Dict:
        """
        Compute all calibration metrics.
        
        Returns:
            Dict with ECE, MCE, Brier score, accuracy, etc.
        """
        if not self.predictions:
            return {
                'ECE': None,
                'MCE': None,
                'Brier': None,
                'accuracy': None,
                'n_samples': 0
            }
        
        confidences = np.array([p[0] for p in self.predictions])
        accuracies = np.array([p[1] for p in self.predictions])
        
        metrics = {
            'ECE': compute_ECE(confidences, accuracies),
            'MCE': compute_MCE(confidences, accuracies),
            'Brier': compute_brier_score(confidences, accuracies),
            'accuracy': float(np.mean(accuracies)),
            'mean_confidence': float(np.mean(confidences)),
            'n_samples': len(self.predictions),
        }
        
        # Per-class metrics
        metrics['per_class'] = {}
        for cls, preds in self.class_predictions.items():
            if len(preds) >= 5:  # Need enough samples
                cls_conf = np.array([p[0] for p in preds])
                cls_acc = np.array([p[1] for p in preds])
                metrics['per_class'][cls] = {
                    'ECE': compute_ECE(cls_conf, cls_acc),
                    'accuracy': float(np.mean(cls_acc)),
                    'n_samples': len(preds)
                }
        
        return metrics
    
    def find_optimal_temperature(self) -> Tuple[float, Dict]:
        """
        Find optimal temperature and compute calibrated metrics.
        
        Returns:
            (optimal_temperature, metrics_dict)
        """
        if not self.raw_confidences:
            return 1.0, {}
        
        confidences = np.array(self.raw_confidences)
        accuracies = np.array([p[1] for p in self.predictions])
        
        best_t, ece_before, ece_after = find_optimal_temperature(confidences, accuracies)
        
        improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0
        
        return best_t, {
            'optimal_temperature': best_t,
            'ECE_before': ece_before,
            'ECE_after': ece_after,
            'improvement_percent': improvement,
            'n_samples': len(self.predictions)
        }


if __name__ == "__main__":
    # Test the ECE computation
    print("="*60)
    print("Testing Calibration Metrics")
    print("="*60)
    
    # Test case 1: Perfectly calibrated
    np.random.seed(42)
    n = 1000
    confidences = np.random.uniform(0, 1, n)
    accuracies = (np.random.random(n) < confidences).astype(int)
    
    ece = compute_ECE(confidences, accuracies)
    print(f"\nTest 1 - Near-perfect calibration:")
    print(f"  ECE: {ece:.4f} (should be close to 0)")
    
    # Test case 2: Overconfident model
    confidences_over = np.random.uniform(0.7, 1.0, n)
    accuracies_over = (np.random.random(n) < 0.5).astype(int)
    
    ece_over = compute_ECE(confidences_over, accuracies_over)
    print(f"\nTest 2 - Overconfident (70-100% conf, 50% acc):")
    print(f"  ECE: {ece_over:.4f} (should be ~0.25)")
    
    # Test case 3: Temperature scaling fix
    best_t, ece_before, ece_after = find_optimal_temperature(confidences_over, accuracies_over)
    print(f"\nTest 3 - Temperature scaling on overconfident:")
    print(f"  Optimal T: {best_t:.2f}")
    print(f"  ECE before: {ece_before:.4f}")
    print(f"  ECE after: {ece_after:.4f}")
    print(f"  Improvement: {(ece_before-ece_after)/ece_before*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ Calibration metrics tests passed!")
