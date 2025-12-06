"""  
Realistic Depth Sensor Simulation Module.

Simulates depth sensor with CHALLENGING noise characteristics:
- Raycast for ground truth geometry
- Significant Gaussian noise (~8-12% at typical distances)
- Random invalid pixels (sensor dropouts)
- Quantization noise

Default parameters simulate difficult conditions:
- Dusty/dirty lens
- Challenging lighting
- Degraded sensor performance

This provides a more realistic challenge for the POMDP
localization system to handle uncertainty.

Noise at 2.5m: ~13cm (5cm base + 4% * 2.5m = ~5 + 10 = 13cm)
Expected error: ~5-10% at typical indoor distances
"""

import numpy as np
import math
from typing import Tuple, Optional, Callable


class RealisticDepthSensor:
    """
    Simulates Intel RealSense D455 depth characteristics.
    
    Usage:
        sensor = RealisticDepthSensor()
        depth, valid, hit = sensor.get_depth_at_pixel(cam_pos, u, v, raycast_func)
    """
    
    def __init__(
        self,
        # Noise parameters - CHALLENGING REALISTIC SCENARIO
        # This simulates a degraded sensor or difficult conditions
        base_noise_std: float = 0.05,         # 5cm base noise (vs 2mm ideal)
        distance_noise_factor: float = 0.04,  # 4% of distance added as noise
        quantization_step: float = 0.005,     # 5mm depth quantization
        
        # Invalid pixel parameters
        random_invalid_prob: float = 0.05,    # 5% random invalid pixels
        edge_invalid_prob: float = 0.3,       # 30% of edge pixels invalid
        min_valid_range: float = 0.4,         # Minimum valid range (D455 spec)
        max_valid_range: float = 10.0,        # Maximum valid range
        
        # Camera parameters
        hfov_deg: float = 60.0,               # Horizontal FOV
        img_width: int = 640,
        img_height: int = 480,
    ):
        """
        Initialize realistic depth sensor.
        
        Args:
            base_noise_std: Base Gaussian noise standard deviation (meters)
            distance_noise_factor: Additional noise as fraction of distance
            quantization_step: Depth quantization step (meters)
            random_invalid_prob: Probability of random pixel dropout
            edge_invalid_prob: Probability of invalid at depth discontinuities
            min_valid_range: Minimum valid depth range (meters)
            max_valid_range: Maximum valid depth range (meters)
            hfov_deg: Horizontal field of view (degrees)
            img_width: Image width in pixels
            img_height: Image height in pixels
        """
        self.base_noise_std = base_noise_std
        self.distance_noise_factor = distance_noise_factor
        self.quantization_step = quantization_step
        self.random_invalid_prob = random_invalid_prob
        self.edge_invalid_prob = edge_invalid_prob
        self.min_valid_range = min_valid_range
        self.max_valid_range = max_valid_range
        self.hfov_deg = hfov_deg
        self.img_width = img_width
        self.img_height = img_height
        
        # Precompute focal length in pixels
        self.focal_length_pixels = img_width / (2 * math.tan(math.radians(hfov_deg / 2)))
        
        # Compute vertical FOV
        self.vfov_deg = 2 * math.degrees(math.atan(
            img_height / (2 * self.focal_length_pixels)
        ))
    
    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Convert pixel coordinates to ray direction.
        
        Assumes camera facing +X direction (Isaac Sim convention).
        LEFT in image corresponds to +Y in world coordinates.
        
        Args:
            u: Horizontal pixel coordinate (0 = left, width = right)
            v: Vertical pixel coordinate (0 = top, height = bottom)
            
        Returns:
            Normalized ray direction [dx, dy, dz]
        """
        hfov = math.radians(self.hfov_deg)
        vfov = math.radians(self.vfov_deg)
        
        # Normalized coordinates (-1 to 1)
        nx = (u - self.img_width / 2) / (self.img_width / 2)
        ny = (self.img_height / 2 - v) / (self.img_height / 2)  # Flip Y for image coords
        
        # Angles from center
        angle_h = nx * (hfov / 2)   # Horizontal angle (yaw)
        angle_v = ny * (vfov / 2)   # Vertical angle (pitch)
        
        # Ray direction (camera facing +X in Isaac Sim)
        # Note: LEFT in image (nx<0) corresponds to +Y in world, so negate dy
        dx = math.cos(angle_v) * math.cos(angle_h)
        dy = -math.cos(angle_v) * math.sin(angle_h)  # Negated for Isaac Sim coords
        dz = math.sin(angle_v)
        
        # Normalize (should already be unit length, but ensure)
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        return np.array([dx/length, dy/length, dz/length])
    
    def add_realistic_noise(self, depth: float) -> float:
        """
        Add realistic noise to a depth measurement.
        
        Models real RealSense D455 noise characteristics:
        - Base Gaussian noise
        - Distance-dependent noise (increases with range)
        - Quantization
        
        Args:
            depth: True depth in meters
            
        Returns:
            Noisy depth measurement
        """
        if depth <= 0:
            return 0.0
        
        # Distance-dependent Gaussian noise
        # Real sensors have noise that increases with distance
        noise_std = self.base_noise_std + self.distance_noise_factor * depth
        noisy_depth = depth + np.random.normal(0, noise_std)
        
        # Quantization (real sensors have discrete depth steps)
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
        """
        Get depth at a pixel location using raycast with optional noise.
        
        Args:
            cam_pos: Camera position in world coordinates [x, y, z]
            u, v: Pixel coordinates
            raycast_func: Function(origin, direction) -> (distance, hit_body)
                         Returns (None, None) if no hit
            add_noise: Whether to add realistic sensor noise
            
        Returns:
            (depth, is_valid, hit_body_name)
            - depth: Distance in meters (0 if invalid)
            - is_valid: Whether the measurement is valid
            - hit_body_name: Name of hit object (or None)
        """
        # Cast ray
        ray_dir = self.pixel_to_ray(u, v)
        distance, hit_body = raycast_func(cam_pos, ray_dir)
        
        # No hit
        if distance is None:
            return 0.0, False, None
        
        # Out of valid range
        if distance < self.min_valid_range or distance > self.max_valid_range:
            return 0.0, False, hit_body
        
        # Random dropout (simulates sensor failures, dust, etc.)
        if add_noise and np.random.random() < self.random_invalid_prob:
            return 0.0, False, hit_body
        
        # Add noise
        if add_noise:
            distance = self.add_realistic_noise(distance)
        
        return distance, True, hit_body
    
    def get_depth_image(
        self, 
        cam_pos: np.ndarray, 
        raycast_func: Callable,
        resolution_scale: float = 1.0,
        add_noise: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Generate a full depth image.
        
        Args:
            cam_pos: Camera position in world coordinates
            raycast_func: Raycast function
            resolution_scale: Scale factor (0.25 = quarter resolution for speed)
            add_noise: Whether to add realistic noise
            verbose: Print progress
            
        Returns:
            Depth image as numpy array (H, W), 0 = invalid
        """
        w = int(self.img_width * resolution_scale)
        h = int(self.img_height * resolution_scale)
        
        depth_img = np.zeros((h, w), dtype=np.float32)
        
        for v in range(h):
            for u in range(w):
                # Map to full resolution coordinates
                u_full = u / resolution_scale
                v_full = v / resolution_scale
                
                depth, valid, _ = self.get_depth_at_pixel(
                    cam_pos, u_full, v_full, raycast_func, add_noise
                )
                depth_img[v, u] = depth if valid else 0.0
            
            if verbose and v % (h // 10) == 0:
                print(f"  Depth image: {100*v/h:.0f}%")
        
        # Add edge invalids (depth discontinuities)
        if add_noise:
            depth_img = self._add_edge_invalids(depth_img)
        
        return depth_img
    
    def _add_edge_invalids(self, depth_img: np.ndarray) -> np.ndarray:
        """Add invalid pixels at depth discontinuities."""
        h, w = depth_img.shape
        result = depth_img.copy()
        
        # Compute gradient magnitude
        grad_x = np.abs(np.diff(depth_img, axis=1, prepend=depth_img[:, :1]))
        grad_y = np.abs(np.diff(depth_img, axis=0, prepend=depth_img[:1, :]))
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Invalidate pixels with high gradient (depth edges)
        edge_threshold = 0.1  # 10cm depth jump
        edge_mask = grad_mag > edge_threshold
        random_mask = np.random.random(depth_img.shape) < self.edge_invalid_prob
        invalid_mask = edge_mask & random_mask
        
        result[invalid_mask] = 0.0
        return result
    
    def get_noise_stats(self, true_depth: float, n_samples: int = 1000) -> dict:
        """
        Get noise statistics for a given true depth.
        
        Args:
            true_depth: True depth to simulate
            n_samples: Number of samples
            
        Returns:
            Dictionary with mean, std, error_percent
        """
        samples = [self.add_realistic_noise(true_depth) for _ in range(n_samples)]
        mean = np.mean(samples)
        std = np.std(samples)
        
        return {
            'true_depth': true_depth,
            'mean': mean,
            'std': std,
            'error_percent': 100 * std / true_depth,
            'bias': mean - true_depth,
        }


def create_isaac_raycast_func(physx_query_interface, max_distance: float = 20.0) -> Callable:
    """
    Create a raycast function compatible with RealisticDepthSensor.
    
    Usage:
        from omni.physx import get_physx_scene_query_interface
        physx_query = get_physx_scene_query_interface()
        raycast_func = create_isaac_raycast_func(physx_query)
        
        depth, valid, body = sensor.get_depth_at_pixel(cam_pos, u, v, raycast_func)
    
    Args:
        physx_query_interface: Isaac Sim physx query interface
        max_distance: Maximum raycast distance
        
    Returns:
        Callable(origin, direction) -> (distance, hit_body)
    """
    def raycast(origin: np.ndarray, direction: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
        origin_tuple = tuple(float(x) for x in origin)
        dir_tuple = tuple(float(x) for x in direction)
        
        hit = physx_query_interface.raycast_closest(origin_tuple, dir_tuple, max_distance)
        
        if hit["hit"]:
            return hit["distance"], hit.get("rigidBody", "unknown")
        return None, None
    
    return raycast


# Demonstration and testing
if __name__ == "__main__":
    print("="*60)
    print("REALISTIC DEPTH SENSOR DEMONSTRATION")
    print("="*60)
    
    sensor = RealisticDepthSensor()
    
    print("\n--- Sensor Parameters ---")
    print(f"  Image size: {sensor.img_width}x{sensor.img_height}")
    print(f"  HFOV: {sensor.hfov_deg}°, VFOV: {sensor.vfov_deg:.1f}°")
    print(f"  Focal length: {sensor.focal_length_pixels:.1f} pixels")
    print(f"  Valid range: {sensor.min_valid_range}m - {sensor.max_valid_range}m")
    print(f"  Base noise: {sensor.base_noise_std*1000:.1f}mm")
    print(f"  Distance noise factor: {sensor.distance_noise_factor*100:.1f}%")
    
    print("\n--- Noise Characteristics ---")
    print("  True Depth → Mean ± Std (Error %)")
    
    for depth in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        stats = sensor.get_noise_stats(depth)
        print(f"  {depth:.1f}m → {stats['mean']:.3f}m ± {stats['std']*100:.1f}cm ({stats['error_percent']:.1f}%)")
    
    print("\n--- Comparison: Ideal vs Challenging Noise ---")
    print("""
  Ideal D455 (clean, good lighting):
    - ~1-2% error at 2-4m
    
  Our CHALLENGING simulation:
    - ~6-10% error at 2-4m
    - Simulates dusty lens, poor lighting, sensor wear
    
  This forces the POMDP to handle real uncertainty!
  Bbox estimation still worse (~40% error)
    """)
    
    print("\n--- Ray Direction Test ---")
    test_pixels = [
        (320, 240, "Center"),
        (0, 240, "Left edge"),
        (640, 240, "Right edge"),
        (320, 0, "Top"),
        (320, 480, "Bottom"),
    ]
    
    for u, v, name in test_pixels:
        ray = sensor.pixel_to_ray(u, v)
        print(f"  {name} ({u}, {v}): [{ray[0]:.3f}, {ray[1]:.3f}, {ray[2]:.3f}]")
    
    print("\n" + "="*60)
    print("✓ Depth sensor module ready!")
    print("="*60)
