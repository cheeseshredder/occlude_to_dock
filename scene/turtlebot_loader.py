"""
TurtleBot3 Burger Loader for Isaac Sim
Loads the actual TurtleBot3 Burger USD model properly.
"""

import numpy as np

try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.prims import XFormPrim
    from pxr import UsdGeom, Gf
    import omni.usd
    import carb
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False


# Possible TurtleBot3 Burger USD paths in Isaac Sim
TURTLEBOT_USD_PATHS = [
    # Isaac Sim 4.0+ paths
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    # Isaac Sim 2023.1.1 paths
    "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    # Local nucleus paths
    "/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    "/NVIDIA/Assets/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    # Built-in sample assets
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
]


def find_turtlebot_usd():
    """
    Find a valid TurtleBot USD path.
    
    Returns:
        str: Valid USD path or None
    """
    import omni.client
    
    for path in TURTLEBOT_USD_PATHS:
        try:
            result, _ = omni.client.stat(path)
            if result == omni.client.Result.OK:
                print(f"  ✓ Found TurtleBot at: {path}")
                return path
        except:
            continue
    
    return None


def load_turtlebot(world, prim_path="/World/turtlebot", position=[0, 0, 0]):
    """
    Load TurtleBot3 Burger into the scene.
    
    Args:
        world: Isaac Sim World instance
        prim_path: Path in USD stage for the robot
        position: Initial position [x, y, z]
        
    Returns:
        robot: Articulation object for the robot, or None if failed
    """
    print("\n--- Loading TurtleBot3 Burger ---")
    
    # Find valid USD path
    usd_path = find_turtlebot_usd()
    
    if usd_path is None:
        print("  ✗ Could not find TurtleBot USD in any known location")
        print("  Trying alternative methods...")
        return load_turtlebot_alternative(world, prim_path, position)
    
    try:
        # Add USD reference to stage
        print(f"  Loading from: {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        
        # Set initial position
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
            print(f"  ✓ TurtleBot placed at {position}")
        
        # Create Articulation wrapper
        robot = world.scene.add(
            Articulation(
                prim_path=prim_path,
                name="turtlebot"
            )
        )
        
        print("  ✓ TurtleBot articulation created")
        return robot
        
    except Exception as e:
        print(f"  ✗ Failed to load TurtleBot: {e}")
        return load_turtlebot_alternative(world, prim_path, position)


def load_turtlebot_alternative(world, prim_path="/World/turtlebot", position=[0, 0, 0]):
    """
    Alternative: Use Robot class directly or search for assets.
    """
    print("\n--- Trying Alternative TurtleBot Loading ---")
    
    try:
        # Method 1: Try using Robot class with asset search
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        
        assets_root = get_assets_root_path()
        if assets_root:
            usd_path = assets_root + "/Isaac/Robots/Turtlebot/turtlebot3_burger.usd"
            print(f"  Trying: {usd_path}")
            
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
            robot = world.scene.add(
                Robot(
                    prim_path=prim_path,
                    name="turtlebot"
                )
            )
            print("  ✓ TurtleBot loaded via Robot class")
            return robot
            
    except Exception as e:
        print(f"  Method 1 failed: {e}")
    
    try:
        # Method 2: Search Isaac Sim assets folder
        from omni.isaac.core.utils.extensions import get_extension_path_from_name
        
        ext_path = get_extension_path_from_name("omni.isaac.robot_assets")
        if ext_path:
            import os
            usd_path = os.path.join(ext_path, "data", "turtlebot3_burger.usd")
            if os.path.exists(usd_path):
                print(f"  Trying: {usd_path}")
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
                
                robot = world.scene.add(
                    Articulation(
                        prim_path=prim_path,
                        name="turtlebot"
                    )
                )
                print("  ✓ TurtleBot loaded from extension")
                return robot
                
    except Exception as e:
        print(f"  Method 2 failed: {e}")
    
    print("  ✗ All TurtleBot loading methods failed")
    print("  → Falling back to simple robot base")
    return None


class TurtleBotController:
    """
    Controller wrapper for TurtleBot3 Burger.
    Provides differential drive control interface.
    """
    
    def __init__(self, robot, world):
        """
        Initialize TurtleBot controller.
        
        Args:
            robot: Articulation object for TurtleBot
            world: Isaac Sim World
        """
        self.robot = robot
        self.world = world
        
        # TurtleBot3 Burger specs
        self.wheel_radius = 0.033  # meters
        self.wheel_base = 0.16    # meters
        
        # Find wheel joint indices
        self.left_wheel_idx = None
        self.right_wheel_idx = None
        self._find_wheel_joints()
        
        print(f"✓ TurtleBotController initialized")
        print(f"  Wheel joints: left={self.left_wheel_idx}, right={self.right_wheel_idx}")
    
    def _find_wheel_joints(self):
        """Find the wheel joint indices in the articulation."""
        try:
            joint_names = self.robot.dof_names
            print(f"  Available joints: {joint_names}")
            
            for i, name in enumerate(joint_names):
                name_lower = name.lower()
                if 'left' in name_lower and 'wheel' in name_lower:
                    self.left_wheel_idx = i
                elif 'right' in name_lower and 'wheel' in name_lower:
                    self.right_wheel_idx = i
                # TurtleBot3 specific joint names
                elif 'wheel_left_joint' in name_lower:
                    self.left_wheel_idx = i
                elif 'wheel_right_joint' in name_lower:
                    self.right_wheel_idx = i
                    
        except Exception as e:
            print(f"  ⚠ Could not find wheel joints: {e}")
    
    def get_pose(self):
        """Get robot pose [x, y, theta]."""
        try:
            pos, quat = self.robot.get_world_pose()
            
            # Convert quaternion to yaw
            w, x, y, z = quat
            theta = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
            
            return np.array([pos[0], pos[1], theta])
        except:
            return np.array([0.0, 0.0, 0.0])
    
    def set_velocity(self, v_linear, v_angular):
        """
        Set robot velocity using differential drive kinematics.
        
        Args:
            v_linear: Linear velocity (m/s)
            v_angular: Angular velocity (rad/s)
        """
        # Calculate wheel velocities
        v_left = (v_linear - (self.wheel_base / 2.0) * v_angular) / self.wheel_radius
        v_right = (v_linear + (self.wheel_base / 2.0) * v_angular) / self.wheel_radius
        
        try:
            if self.left_wheel_idx is not None and self.right_wheel_idx is not None:
                # Set specific wheel joints
                velocities = np.zeros(self.robot.num_dof)
                velocities[self.left_wheel_idx] = v_left
                velocities[self.right_wheel_idx] = v_right
                self.robot.set_joint_velocities(velocities)
            else:
                # Fallback: assume first two DOFs are wheels
                self.robot.set_joint_velocities(np.array([v_left, v_right]))
                
        except Exception as e:
            print(f"⚠ Error setting velocity: {e}")
    
    def stop(self):
        """Stop the robot."""
        self.set_velocity(0.0, 0.0)


# Test function
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING: TurtleBot3 Burger Loader")
    print("=" * 60)
    
    if not ISAAC_SIM_AVAILABLE:
        print("\n❌ This test requires Isaac Sim")
        print("   Run using: isaac-sim-python scene/turtlebot_loader.py")
        exit(1)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from omni.isaac.core import World
        
        # Create world
        print("\nCreating world...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        
        # Load TurtleBot
        robot = load_turtlebot(world, position=[0, 0, 0])
        
        if robot is not None:
            # Initialize physics
            print("\nInitializing physics...")
            world.reset()
            
            # Create controller
            controller = TurtleBotController(robot, world)
            
            # Test movement
            print("\nTesting movement...")
            controller.set_velocity(0.1, 0.0)  # Forward
            
            for _ in range(100):
                world.step(render=True)
            
            pose = controller.get_pose()
            print(f"Pose after forward: {pose}")
            
            controller.stop()
            
            print("\n✅ TurtleBot loaded and working!")
            
            # Keep window open
            print("\nKeep window open - close Isaac Sim to exit")
            while simulation_app.is_running():
                world.step(render=True)
        else:
            print("\n❌ TurtleBot loading failed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()
