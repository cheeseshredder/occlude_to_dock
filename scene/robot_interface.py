"""
Isaac Sim Robot Interface for TurtleBot3 Burger
RGB-only approach - depth estimated from detections
"""

import numpy as np
from typing import Tuple, Optional, Dict
import os


class IsaacSimRobot:
    """
    Interface for TurtleBot3 Burger in Isaac Sim.
    
    Provides:
    - Differential drive control
    - Pose estimation  
    - RGB from viewport capture (depth estimated from perception)
    """
    
    # TurtleBot3 Burger specs
    WHEEL_RADIUS = 0.033  # meters
    WHEEL_BASE = 0.16    # meters
    
    def __init__(self, robot, world, camera_path: str, objects: Optional[Dict] = None):
        """
        Initialize robot interface.
        
        Args:
            robot: WheeledRobot instance from Isaac Sim
            world: World instance
            camera_path: Path to robot camera prim
            objects: Scene objects (optional, for ground truth)
        """
        self.robot = robot
        self.world = world
        self.camera_path = camera_path
        self.objects = objects or {}
        
        # Setup viewport for RGB capture
        self.viewport = None
        self._setup_viewport()
        
        # Import Isaac Sim types
        from omni.isaac.core.utils.types import ArticulationAction
        self.ArticulationAction = ArticulationAction
        
        print(f"✓ IsaacSimRobot initialized")
        print(f"  Camera: {camera_path}")
    
    def _setup_viewport(self):
        """Setup viewport for RGB capture."""
        try:
            from omni.kit.viewport.utility import get_active_viewport
            self.viewport = get_active_viewport()
            if self.camera_path:
                self.viewport.set_active_camera(self.camera_path)
                print(f"  Viewport switched to robot camera")
        except Exception as e:
            print(f"⚠ Viewport setup warning: {e}")
            self.viewport = None
    
    def refresh_viewport(self):
        """Re-sync viewport to camera (call after warm-up)."""
        try:
            from omni.kit.viewport.utility import get_active_viewport
            self.viewport = get_active_viewport()
            if self.camera_path:
                self.viewport.set_active_camera(self.camera_path)
                # Step a few times to let the viewport update
                for _ in range(20):
                    self.world.step(render=True)
                print(f"  ✓ Viewport refreshed to: {self.camera_path}")
        except Exception as e:
            print(f"⚠ Viewport refresh warning: {e}")
    
    def get_pose(self) -> Tuple[float, float, float]:
        """
        Get robot pose (x, y, yaw).
        
        Returns:
            (x, y, yaw) in world frame, yaw in radians
        """
        pos, quat = self.robot.get_world_pose()
        
        # Convert quaternion [w, x, y, z] to yaw
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return float(pos[0]), float(pos[1]), float(yaw)
    
    def set_velocity(self, linear: float, angular: float):
        """
        Set robot velocity using differential drive.
        
        Args:
            linear: Forward velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        # Differential drive: v_left = (v - w*L/2)/r, v_right = (v + w*L/2)/r
        v_left = (linear - angular * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        v_right = (linear + angular * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        
        self.robot.apply_wheel_actions(
            self.ArticulationAction(joint_velocities=np.array([v_left, v_right]))
        )
    
    def stop(self):
        """Stop the robot."""
        self.set_velocity(0.0, 0.0)
    
    def get_rgb(self) -> np.ndarray:
        """
        Capture RGB image from robot camera.
        
        Returns:
            rgb: (480, 640, 3) uint8 array
        """
        # Step MORE times to ensure rendering completes (was 3, now 10)
        for _ in range(10):
            self.world.step(render=True)
        
        if self.viewport is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file
            from PIL import Image
            import tempfile
            import time
            
            # Capture to temp file
            temp_path = os.path.join(tempfile.gettempdir(), "robot_rgb.png")
            capture_viewport_to_file(self.viewport, temp_path)
            
            # Wait for file to be written (critical for proper capture)
            time.sleep(0.1)
            
            if os.path.exists(temp_path):
                img = Image.open(temp_path)
                img = img.resize((640, 480), Image.LANCZOS)
                rgb = np.array(img)
                
                if rgb.shape[-1] == 4:  # RGBA -> RGB
                    rgb = rgb[:, :, :3]
                
                os.remove(temp_path)
                return rgb.astype(np.uint8)
                
        except Exception as e:
            print(f"⚠ RGB capture error: {e}")
        
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def step(self, render: bool = True):
        """Step the simulation."""
        self.world.step(render=render)
    
    def get_object_ground_truth(self) -> Dict:
        """
        Get ground truth object positions (for evaluation only).
        
        Returns:
            Dict mapping object names to world positions
        """
        gt = {}
        for name, obj in self.objects.items():
            try:
                pos, _ = obj.get_world_pose()
                gt[name] = [float(pos[0]), float(pos[1]), float(pos[2])]
            except:
                pass
        return gt


class MockRobot:
    """
    Mock robot for testing without Isaac Sim.
    """
    
    def __init__(self, start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.x, self.y, self.yaw = start_pose
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.dt = 0.05  # 20 Hz
        
        # Mock scene
        self.objects_gt = {
            'charger': [2.8, 0.0, 0.25],
            'plant': [2.0, 0.0, 0.5],
            'distractor_1': [2.0, 1.5, 0.225],
            'distractor_2': [2.0, -1.5, 0.225],
        }
    
    def get_pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.yaw
    
    def set_velocity(self, linear: float, angular: float):
        self.linear_vel = linear
        self.angular_vel = angular
    
    def stop(self):
        self.set_velocity(0.0, 0.0)
    
    def get_rgb(self) -> np.ndarray:
        """Generate mock RGB image."""
        # Simple colored background
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :] = [135, 206, 235]  # Sky blue
        
        # Draw floor (bottom half)
        rgb[300:, :] = [100, 149, 237]  # Cornflower blue
        
        return rgb
    
    def step(self, render: bool = True):
        """Update pose based on velocity."""
        self.x += self.linear_vel * np.cos(self.yaw) * self.dt
        self.y += self.linear_vel * np.sin(self.yaw) * self.dt
        self.yaw += self.angular_vel * self.dt
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))
    
    def get_object_ground_truth(self) -> Dict:
        return self.objects_gt.copy()


def create_robot(mode: str = 'mock', **kwargs):
    """
    Factory function to create robot interface.
    
    Args:
        mode: 'mock' or 'isaac_sim'
        **kwargs: Additional arguments
        
    Returns:
        Robot interface instance
    """
    if mode == 'mock':
        start_pose = kwargs.get('start_pose', (0.0, 0.0, 0.0))
        return MockRobot(start_pose=start_pose)
    
    elif mode == 'isaac_sim':
        return IsaacSimRobot(
            robot=kwargs['robot'],
            world=kwargs['world'],
            camera_path=kwargs.get('camera_path', ''),
            objects=kwargs.get('objects', {})
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
