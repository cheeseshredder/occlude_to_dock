"""
Utility functions used across the Occlude-to-Dock system.
"""

import numpy as np
import yaml
from typing import Tuple, List


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def angle_normalize(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get all grid cells along line from start to end using Bresenham's algorithm.
    
    Args:
        start: (i, j) starting grid cell
        end: (i, j) ending grid cell
        
    Returns:
        List of (i, j) tuples representing cells along the line
    """
    cells = []
    
    x0, y0 = start
    x1, y1 = end
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        cells.append((x0, y0))
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return cells


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union of two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IOU value in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid function: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + np.exp(-x))


def logit_inverse(p: float) -> float:
    """
    Convert probability to logit: log(p / (1-p))
    
    Args:
        p: Probability in (0, 1)
        
    Returns:
        Logit value
    """
    p = np.clip(p, 1e-8, 1 - 1e-8)  # Avoid division by zero
    return np.log(p / (1 - p))


def pixel_to_world(
    u: float, 
    v: float, 
    depth: float,
    robot_pose: np.ndarray,
    camera_params: dict
) -> np.ndarray:
    """
    Project pixel coordinates to 3D world coordinates.
    
    Args:
        u, v: Pixel coordinates
        depth: Depth value in meters
        robot_pose: Robot pose [x, y, z, qx, qy, qz, qw] or [x, y, theta]
        camera_params: Camera intrinsics and extrinsics
        
    Returns:
        3D point in world frame [x, y, z]
    """
    # Get camera intrinsics
    K = camera_params['intrinsics']
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Pixel to camera frame
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth
    p_camera = np.array([x_c, y_c, z_c, 1.0])
    
    # Camera to robot frame
    T_C_R = camera_params['extrinsics']  # 4×4 transform
    p_robot = T_C_R @ p_camera
    
    # Robot to world frame
    # Simple 2D case: robot_pose = [x, y, theta]
    if len(robot_pose) == 3:
        x_r, y_r, theta = robot_pose
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x_w = x_r + cos_theta * p_robot[0] - sin_theta * p_robot[1]
        y_w = y_r + sin_theta * p_robot[0] + cos_theta * p_robot[1]
        z_w = p_robot[2]
        
        return np.array([x_w, y_w, z_w])
    else:
        # Full 3D transform (not implemented yet)
        raise NotImplementedError("Full 3D transform not yet implemented")


def create_camera_intrinsics(resolution: Tuple[int, int], fov: float) -> np.ndarray:
    """
    Create camera intrinsics matrix from resolution and FOV.
    
    Args:
        resolution: (width, height) in pixels
        fov: Field of view in degrees
        
    Returns:
        3×3 intrinsics matrix K
    """
    width, height = resolution
    fov_rad = np.radians(fov)
    
    # Compute focal length
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    
    # Principal point (image center)
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def create_camera_extrinsics(position: List[float], orientation: List[float]) -> np.ndarray:
    """
    Create camera extrinsics (robot to camera transform).
    
    Args:
        position: [x, y, z] relative to robot base
        orientation: [roll, pitch, yaw] in radians
        
    Returns:
        4×4 transformation matrix
    """
    x, y, z = position
    roll, pitch, yaw = orientation
    
    # Rotation matrix (ZYX Euler angles)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test angle normalization
    assert np.isclose(angle_normalize(3.5 * np.pi), -0.5 * np.pi)
    print("✓ angle_normalize")
    
    # Test IOU
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_iou(box1, box2)
    assert np.isclose(iou, 0.142857, atol=1e-5)
    print("✓ compute_iou")
    
    # Test sigmoid
    assert np.isclose(sigmoid(0), 0.5)
    assert np.isclose(sigmoid(100), 1.0)
    print("✓ sigmoid")
    
    # Test Bresenham
    line = bresenham_line((0, 0), (5, 5))
    assert len(line) == 6  # Including start and end
    print("✓ bresenham_line")
    
    print("\nAll utility tests passed!")
