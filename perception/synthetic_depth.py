"""
Synthetic Depth Generator for POMDP Project
Computes depth from robot pose and known object positions.
Adds realistic noise to simulate real depth sensor behavior.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class SyntheticDepthGenerator:
    """
    Generates synthetic depth images from ground truth object positions.
    
    This is valid for POMDP because:
    1. Real depth sensors are noisy anyway
    2. We control the noise model (important for belief updates)
    3. Allows testing without broken Replicator
    """
    
    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        fov_horizontal: float = 60.0,  # degrees
        fov_vertical: float = 45.0,    # degrees
        max_depth: float = 10.0,       # meters
        noise_std: float = 0.02,       # depth noise standard deviation
        missing_data_prob: float = 0.01,  # probability of missing depth
    ):
        self.width = image_width
        self.height = image_height
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.max_depth = max_depth
        self.noise_std = noise_std
        self.missing_data_prob = missing_data_prob
        
        # Precompute ray directions for each pixel
        self._precompute_rays()
    
    def _precompute_rays(self):
        """Precompute ray directions for each pixel."""
        # Create pixel grid
        u = np.arange(self.width)
        v = np.arange(self.height)
        uu, vv = np.meshgrid(u, v)
        
        # Convert to normalized coordinates (-1 to 1)
        u_norm = (uu - self.width / 2) / (self.width / 2)
        v_norm = (vv - self.height / 2) / (self.height / 2)
        
        # Convert to angles
        theta = u_norm * (self.fov_h / 2)  # horizontal angle
        phi = v_norm * (self.fov_v / 2)    # vertical angle
        
        # Ray directions in camera frame (looking along +X)
        self.ray_dirs = np.stack([
            np.cos(theta) * np.cos(phi),  # forward (X)
            np.sin(theta),                 # right (Y)
            -np.sin(phi)                   # up (Z)
        ], axis=-1)
    
    def generate_depth(
        self,
        robot_pose: Tuple[float, float, float],  # (x, y, yaw)
        objects: Dict[str, dict],  # {name: {'position': [x,y,z], 'size': [w,d,h]}}
        floor_height: float = 0.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Generate synthetic depth image.
        
        Args:
            robot_pose: (x, y, yaw) of robot in world frame
            objects: Dictionary of objects with positions and sizes
            floor_height: Height of floor plane
            add_noise: Whether to add realistic noise
            
        Returns:
            depth: (H, W) float32 array of depths in meters
        """
        x, y, yaw = robot_pose
        camera_height = 0.2  # Camera is 20cm above ground
        
        # Initialize depth to max (sky/far)
        depth = np.full((self.height, self.width), self.max_depth, dtype=np.float32)
        
        # Transform ray directions to world frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Rotation matrix (robot frame to world frame)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Camera position in world frame
        cam_pos = np.array([x, y, camera_height])
        
        # Ray directions in world frame
        rays_world = np.einsum('ij,hwj->hwi', R, self.ray_dirs)
        
        # Intersect with floor plane
        # Floor at z = floor_height
        # Ray: p = cam_pos + t * ray_dir
        # Floor: z = floor_height
        # t = (floor_height - cam_pos[2]) / ray_dir[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_floor = (floor_height - cam_pos[2]) / rays_world[:, :, 2]
            t_floor = np.where(t_floor > 0, t_floor, self.max_depth)
            depth = np.minimum(depth, t_floor)
        
        # Intersect with each object (approximated as axis-aligned boxes)
        for obj_name, obj_info in objects.items():
            obj_pos = np.array(obj_info['position'])
            obj_size = np.array(obj_info.get('size', [0.5, 0.5, 1.0]))
            
            # Box bounds
            box_min = obj_pos - obj_size / 2
            box_max = obj_pos + obj_size / 2
            
            # Ray-box intersection (simplified)
            t_obj = self._ray_box_intersection(cam_pos, rays_world, box_min, box_max)
            depth = np.minimum(depth, t_obj)
        
        # Intersect with walls (if room bounds provided)
        room_bounds = [[-3.0, 3.0], [-3.0, 3.0]]  # Default 6x6 room
        depth = self._intersect_walls(cam_pos, rays_world, depth, room_bounds)
        
        # Add noise
        if add_noise:
            # Depth-dependent noise (further = more noise)
            noise_scale = self.noise_std * (1 + depth / self.max_depth)
            noise = np.random.normal(0, noise_scale)
            depth = depth + noise
            
            # Random missing data (like real depth sensors)
            missing_mask = np.random.random((self.height, self.width)) < self.missing_data_prob
            depth[missing_mask] = 0.0  # 0 = invalid depth
        
        # Clip to valid range
        depth = np.clip(depth, 0, self.max_depth)
        
        return depth.astype(np.float32)
    
    def _ray_box_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dirs: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> np.ndarray:
        """Compute ray-box intersection distances."""
        # Slab method for AABB intersection
        t_min = np.full((self.height, self.width), -np.inf)
        t_max = np.full((self.height, self.width), np.inf)
        
        for i in range(3):
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (box_min[i] - ray_origin[i]) / ray_dirs[:, :, i]
                t2 = (box_max[i] - ray_origin[i]) / ray_dirs[:, :, i]
            
            t_near = np.minimum(t1, t2)
            t_far = np.maximum(t1, t2)
            
            t_min = np.maximum(t_min, t_near)
            t_max = np.minimum(t_max, t_far)
        
        # Valid intersection if t_min < t_max and t_min > 0
        valid = (t_min < t_max) & (t_min > 0)
        result = np.where(valid, t_min, self.max_depth)
        
        return result
    
    def _intersect_walls(
        self,
        cam_pos: np.ndarray,
        rays_world: np.ndarray,
        depth: np.ndarray,
        room_bounds: list,
    ) -> np.ndarray:
        """Intersect rays with room walls."""
        # Four walls as planes
        walls = [
            (np.array([1, 0, 0]), room_bounds[0][1]),   # +X wall (East)
            (np.array([-1, 0, 0]), -room_bounds[0][0]), # -X wall (West)
            (np.array([0, 1, 0]), room_bounds[1][1]),   # +Y wall (North)
            (np.array([0, -1, 0]), -room_bounds[1][0]), # -Y wall (South)
        ]
        
        for normal, d in walls:
            # Plane equation: normal · p = d
            # Ray: p = cam_pos + t * ray_dir
            # t = (d - normal · cam_pos) / (normal · ray_dir)
            denom = np.sum(rays_world * normal, axis=-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (d - np.dot(normal, cam_pos)) / denom
                t = np.where((t > 0) & (denom > 0), t, self.max_depth)
            depth = np.minimum(depth, t)
        
        return depth


def create_depth_generator(config: Optional[dict] = None) -> SyntheticDepthGenerator:
    """Factory function to create depth generator with config."""
    if config is None:
        config = {}
    
    return SyntheticDepthGenerator(
        image_width=config.get('image_width', 640),
        image_height=config.get('image_height', 480),
        fov_horizontal=config.get('fov_horizontal', 60.0),
        fov_vertical=config.get('fov_vertical', 45.0),
        max_depth=config.get('max_depth', 10.0),
        noise_std=config.get('depth_noise_std', 0.02),
        missing_data_prob=config.get('missing_data_prob', 0.01),
    )


# Quick test
if __name__ == "__main__":
    print("Testing Synthetic Depth Generator...")
    
    gen = SyntheticDepthGenerator()
    
    # Robot at origin, facing +X
    robot_pose = (0.0, 0.0, 0.0)
    
    # Objects in scene
    objects = {
        'charger': {'position': [2.8, 0.0, 0.25], 'size': [0.4, 0.3, 0.5]},
        'plant': {'position': [2.0, 0.0, 0.5], 'size': [0.6, 0.6, 1.0]},
        'distractor_1': {'position': [2.0, 1.5, 0.225], 'size': [0.35, 0.25, 0.45]},
    }
    
    depth = gen.generate_depth(robot_pose, objects)
    
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    print(f"Mean depth: {depth.mean():.2f} meters")
    
    # Save visualization
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(depth, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title('Synthetic Depth Image')
        plt.savefig('synthetic_depth_test.png')
        print("Saved: synthetic_depth_test.png")
    except ImportError:
        print("matplotlib not available for visualization")
    
    print("✅ Test complete!")
