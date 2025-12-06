"""     
Isaac Sim Scene Setup - TurtleBot3 Burger with Robot Camera + AprilTag
"""

import numpy as np
import yaml
import os

try:
    from omni.isaac.kit import SimulationApp
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, VisualCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.wheeled_robots.robots import WheeledRobot
    from pxr import UsdLux, UsdGeom, Gf, UsdShade, Sdf
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False


# AprilTag 36h11 patterns (10x10 including border)
# 1 = white, 0 = black
# Official patterns from: https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/

# ID=0 (charger face tag)
APRILTAG_36H11_ID0 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8)

# Wall tags use the SAME pattern as ID=0 (which works!)
# We distinguish them by their known positions
# TWO wall tags offset to sides so they're visible AROUND the charger
WALL_TAG_POSITIONS = {
    # Wall tag NORTH (+Y side) - visible when approaching from -Y
    'north': (2.95, 0.40, 0.30),  # Offset +Y, slightly higher than camera
    # Wall tag SOUTH (-Y side) - visible when approaching from +Y  
    'south': (2.95, -0.40, 0.30),  # Offset -Y, slightly higher than camera
}


class OccludeToDockScene:
    
    def __init__(self, config_path='config/scene_partial.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.world = None
        self.robot = None
        self.camera_path = None
        self.objects = {}
        self.apriltag_path = None
        
    def setup(self, seed=42, use_apriltag=True):
        if not ISAAC_SIM_AVAILABLE:
            raise RuntimeError("Isaac Sim required")
        
        np.random.seed(seed)
        
        print("=" * 60)
        print("Setting up Occlude-to-Dock Scene (TurtleBot + Camera + AprilTag)")
        print("=" * 60)
        
        print("\n[1/8] Creating world...")
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        print("[2/8] Creating room boundaries...")
        self._create_room()
        
        print("[3/8] Configuring lighting...")
        self._setup_lighting()
        
        print("[4/8] Spawning objects...")
        self._spawn_objects()
        
        if use_apriltag:
            print("[5/8] Adding AprilTags (charger + wall)...")
            self._create_apriltag_on_charger()
            # Wall tags provide FRONTAL views when approaching from sides!
            self._create_wall_apriltags()
        else:
            print("[5/8] Skipping AprilTags (use_apriltag=False)")
        
        print("[6/8] Spawning TurtleBot3 Burger...")
        self._spawn_turtlebot()
        
        print("[7/8] Creating robot camera...")
        self._create_robot_camera()
        
        print("[8/8] Initializing physics...")
        self.world.reset()
        
        print("✅ Scene setup complete!")
        print("=" * 60)
        
        return {
            'world': self.world,
            'robot': self.robot,
            'camera_path': self.camera_path,
            'objects': self.objects,
            'apriltag_path': self.apriltag_path,
            'config': self.config
        }
    
    def _create_room(self):
        """Create room with walls on the ground"""
        room_size = self.config.get('room_size', [6.0, 6.0])
        wall_height = 2.0
        wall_thickness = 0.1
        
        half_x = room_size[0] / 2
        half_y = room_size[1] / 2
        
        # North wall (+Y)
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/walls/north",
                name="wall_north",
                position=np.array([0.0, half_y, wall_height/2]),
                scale=np.array([room_size[0], wall_thickness, wall_height]),
                size=1.0,
                color=np.array([0.9, 0.9, 0.9]),
            )
        )
        
        # South wall (-Y)
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/walls/south",
                name="wall_south",
                position=np.array([0.0, -half_y, wall_height/2]),
                scale=np.array([room_size[0], wall_thickness, wall_height]),
                size=1.0,
                color=np.array([0.9, 0.9, 0.9]),
            )
        )
        
        # East wall (+X)
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/walls/east",
                name="wall_east",
                position=np.array([half_x, 0.0, wall_height/2]),
                scale=np.array([wall_thickness, room_size[1], wall_height]),
                size=1.0,
                color=np.array([0.9, 0.9, 0.9]),
            )
        )
        
        # West wall (-X)
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/walls/west",
                name="wall_west",
                position=np.array([-half_x, 0.0, wall_height/2]),
                scale=np.array([wall_thickness, room_size[1], wall_height]),
                size=1.0,
                color=np.array([0.9, 0.9, 0.9]),
            )
        )
        
        print(f"  ✓ Room: {room_size[0]}m x {room_size[1]}m")
    
    def _setup_lighting(self):
        stage = self.world.stage
        
        distant_light = UsdLux.DistantLight.Define(stage, "/World/sunLight")
        distant_light.CreateIntensityAttr(1000.0)
        
        dome_light = UsdLux.DomeLight.Define(stage, "/World/domeLight")
        dome_light.CreateIntensityAttr(300.0)
        
        print("  ✓ Lighting configured")
    
    def _spawn_objects(self):
        """Spawn objects ON THE GROUND with collision enabled for raycasts."""
        from pxr import UsdPhysics
        
        objects_config = self.config['objects']
        stage = self.world.stage
        
        # Charger (GREEN target)
        charger_cfg = objects_config['charger']
        charger_size = charger_cfg['size']
        charger_pos = charger_cfg['position']
        
        self.objects['charger'] = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/charger",
                name="charger",
                position=np.array([charger_pos[0], charger_pos[1], charger_size[2]/2]),
                scale=np.array([charger_size[0], charger_size[1], charger_size[2]]),
                size=1.0,
                color=np.array([0.0, 0.8, 0.0]),
            )
        )
        charger_prim = stage.GetPrimAtPath("/World/charger")
        if not charger_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(charger_prim)
        print(f"  ✓ Charger at [{charger_pos[0]}, {charger_pos[1]}] (GREEN)")
        
        # Plant (RED occluder)
        plant_cfg = objects_config['plant']
        plant_height = plant_cfg['height']
        plant_radius = plant_cfg['radius']
        plant_pos = plant_cfg['position']
        
        self.objects['plant'] = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/plant",
                name="plant",
                position=np.array([plant_pos[0], plant_pos[1], plant_height/2]),
                scale=np.array([plant_radius*2, plant_radius*2, plant_height]),
                size=1.0,
                color=np.array([0.8, 0.0, 0.0]),
            )
        )
        plant_prim = stage.GetPrimAtPath("/World/plant")
        if not plant_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(plant_prim)
        print(f"  ✓ Plant at [{plant_pos[0]}, {plant_pos[1]}] (RED - occluder)")
        
        # Distractors (GRAY)
        for i, distractor_key in enumerate(['distractor_1', 'distractor_2']):
            dist_cfg = objects_config[distractor_key]
            dist_size = dist_cfg.get('size', [0.35, 0.25, 0.45])
            dist_pos = dist_cfg['position']
            
            self.objects[distractor_key] = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/distractor_{i}",
                    name=distractor_key,
                    position=np.array([dist_pos[0], dist_pos[1], dist_size[2]/2]),
                    scale=np.array([dist_size[0], dist_size[1], dist_size[2]]),
                    size=1.0,
                    color=np.array([0.5, 0.5, 0.5]),
                )
            )
            dist_prim = stage.GetPrimAtPath(f"/World/distractor_{i}")
            if not dist_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(dist_prim)
            print(f"  ✓ {distractor_key} at [{dist_pos[0]}, {dist_pos[1]}] (GRAY)")
    
    def _create_apriltag_on_charger(self, tag_size=0.06):
        """
        Create AprilTag pattern on THREE faces of charger (-X, +Y, -Y).
        
        Uses SMALLER tags (6cm) for close-range docking detection.
        Wall tags remain at 15cm for long-range navigation.
        
        This ensures the tag is visible from any approach direction:
        - -X face: Direct approach from robot start position
        - +Y face: If robot goes around plant on the +Y (north) side
        - -Y face: If robot goes around plant on the -Y (south) side
        
        Args:
            tag_size: Total size of the tag in meters (default 6cm for close-range)
        """
        objects_config = self.config['objects']
        charger_cfg = objects_config['charger']
        charger_pos = charger_cfg['position']
        charger_size = charger_cfg['size']
        
        # Tag center height
        tag_z = charger_size[2] / 2
        
        # Cell dimensions
        cell_size = tag_size / 10.0  # 10x10 grid
        cell_thickness = 0.002  # 2mm thick
        
        stage = self.world.stage
        
        # Create parent xform for the tags
        self.apriltag_path = "/World/apriltag"
        apriltag_xform = UsdGeom.Xform.Define(stage, self.apriltag_path)
        
        pattern = APRILTAG_36H11_ID0
        total_black_cubes = 0
        
        # ============================================================
        # FACE 1: -X face (front, facing robot's main approach)
        # ============================================================
        tag_x_minus = charger_pos[0] - charger_size[0] / 2 - 0.002
        tag_y_center = charger_pos[1]
        
        # White background
        self.world.scene.add(
            FixedCuboid(
                prim_path=f"{self.apriltag_path}/white_bg_minus_x",
                name=f"apriltag_white_minus_x",
                position=np.array([tag_x_minus, tag_y_center, tag_z]),
                scale=np.array([cell_thickness, tag_size * 1.1, tag_size * 1.1]),
                size=1.0,
                color=np.array([1.0, 1.0, 1.0]),
            )
        )
        
        # Black cells for -X face
        # When viewed from -X: robot's LEFT is +Y, robot's RIGHT is -Y
        # col=0 should appear on LEFT, so col=0 -> high Y
        black_x_minus = tag_x_minus - 0.002
        for row in range(10):
            for col in range(10):
                if pattern[row, col] == 0:
                    cell_y = tag_y_center + (4.5 - col) * cell_size  # col 0 on LEFT (+Y side)
                    cell_z = tag_z + (4.5 - row) * cell_size
                    self.world.scene.add(
                        FixedCuboid(
                            prim_path=f"{self.apriltag_path}/black_minus_x_{row}_{col}",
                            name=f"apriltag_black_minus_x_{row}_{col}",
                            position=np.array([black_x_minus, cell_y, cell_z]),
                            scale=np.array([cell_thickness, cell_size * 0.95, cell_size * 0.95]),
                            size=1.0,
                            color=np.array([0.0, 0.0, 0.0]),
                        )
                    )
                    total_black_cubes += 1
        
        # ============================================================
        # FACE 2: +Y face (left side when viewed from -X)
        # ============================================================
        tag_y_plus = charger_pos[1] + charger_size[1] / 2 + 0.002
        tag_x_center = charger_pos[0]
        
        # White background
        self.world.scene.add(
            FixedCuboid(
                prim_path=f"{self.apriltag_path}/white_bg_plus_y",
                name=f"apriltag_white_plus_y",
                position=np.array([tag_x_center, tag_y_plus, tag_z]),
                scale=np.array([tag_size * 1.1, cell_thickness, tag_size * 1.1]),
                size=1.0,
                color=np.array([1.0, 1.0, 1.0]),
            )
        )
        
        # Black cells for +Y face
        # WORKING PATTERN: use negative sign (mirrored)
        black_y_plus = tag_y_plus + 0.002
        for row in range(10):
            for col in range(10):
                if pattern[row, col] == 0:
                    cell_x = tag_x_center - (4.5 - col) * cell_size  # Mirrored (WORKING)
                    cell_z = tag_z + (4.5 - row) * cell_size
                    self.world.scene.add(
                        FixedCuboid(
                            prim_path=f"{self.apriltag_path}/black_plus_y_{row}_{col}",
                            name=f"apriltag_black_plus_y_{row}_{col}",
                            position=np.array([cell_x, black_y_plus, cell_z]),
                            scale=np.array([cell_size * 0.95, cell_thickness, cell_size * 0.95]),
                            size=1.0,
                            color=np.array([0.0, 0.0, 0.0]),
                        )
                    )
                    total_black_cubes += 1
        
        # ============================================================
        # FACE 3: -Y face (right side when viewed from -X)
        # ============================================================
        tag_y_minus = charger_pos[1] - charger_size[1] / 2 - 0.002
        
        # White background
        self.world.scene.add(
            FixedCuboid(
                prim_path=f"{self.apriltag_path}/white_bg_minus_y",
                name=f"apriltag_white_minus_y",
                position=np.array([tag_x_center, tag_y_minus, tag_z]),
                scale=np.array([tag_size * 1.1, cell_thickness, tag_size * 1.1]),
                size=1.0,
                color=np.array([1.0, 1.0, 1.0]),
            )
        )
        
        # Black cells for -Y face
        # WORKING PATTERN: use positive sign (normal)
        black_y_minus = tag_y_minus - 0.002
        for row in range(10):
            for col in range(10):
                if pattern[row, col] == 0:
                    cell_x = tag_x_center + (4.5 - col) * cell_size  # Normal (WORKING)
                    cell_z = tag_z + (4.5 - row) * cell_size
                    self.world.scene.add(
                        FixedCuboid(
                            prim_path=f"{self.apriltag_path}/black_minus_y_{row}_{col}",
                            name=f"apriltag_black_minus_y_{row}_{col}",
                            position=np.array([cell_x, black_y_minus, cell_z]),
                            scale=np.array([cell_size * 0.95, cell_thickness, cell_size * 0.95]),
                            size=1.0,
                            color=np.array([0.0, 0.0, 0.0]),
                        )
                    )
                    total_black_cubes += 1
        
        print(f"  ✓ AprilTag ID=0 on THREE faces of charger (-X, +Y, -Y)")
        print(f"    Size: {tag_size*100:.0f}cm x {tag_size*100:.0f}cm each")
        print(f"    Pattern: {total_black_cubes} black cells total")
        
        self.objects['apriltag'] = self.apriltag_path
    
    def _create_wall_apriltags(self, tag_size=0.15):
        """
        Create AprilTag markers on the BACK WALL, offset to sides of charger.
        
        Uses LARGER tags (15cm) for long-range navigation detection.
        Charger tags are smaller (6cm) for close-range docking.
        
        Uses the SAME ID=0 pattern as the charger tags (which works!).
        TWO wall tags positioned to be visible AROUND the charger:
        - North tag (+Y): visible when robot approaches from -Y side
        - South tag (-Y): visible when robot approaches from +Y side
        
        Args:
            tag_size: Size of tag in meters (default 15cm for long-range)
        """
        stage = self.world.stage
        cell_size = tag_size / 10.0  # 10x10 grid
        cell_thickness = 0.002
        
        # Create parent xform for wall tags
        wall_tags_path = "/World/wall_apriltags"
        UsdGeom.Xform.Define(stage, wall_tags_path)
        
        # Use ID=0 pattern (known to work!)
        pattern = APRILTAG_36H11_ID0
        
        total_black_cubes = 0
        
        # Create both wall tags
        for tag_name, (tag_x, tag_y, tag_z) in WALL_TAG_POSITIONS.items():
            tag_path = f"{wall_tags_path}/{tag_name}"
            
            # White background (on YZ plane, facing -X)
            self.world.scene.add(
                FixedCuboid(
                    prim_path=f"{tag_path}/white_bg",
                    name=f"wall_tag_{tag_name}_white",
                    position=np.array([tag_x, tag_y, tag_z]),
                    scale=np.array([cell_thickness, tag_size * 1.1, tag_size * 1.1]),
                    size=1.0,
                    color=np.array([1.0, 1.0, 1.0]),
                )
            )
            
            # Black cells for -X face (facing robot approach)
            black_x = tag_x - 0.002
            for row in range(10):
                for col in range(10):
                    if pattern[row, col] == 0:
                        cell_y = tag_y + (4.5 - col) * cell_size
                        cell_z = tag_z + (4.5 - row) * cell_size
                        self.world.scene.add(
                            FixedCuboid(
                                prim_path=f"{tag_path}/black_{row}_{col}",
                                name=f"wall_tag_{tag_name}_black_{row}_{col}",
                                position=np.array([black_x, cell_y, cell_z]),
                                scale=np.array([cell_thickness, cell_size * 0.95, cell_size * 0.95]),
                                size=1.0,
                                color=np.array([0.0, 0.0, 0.0]),
                            )
                        )
                        total_black_cubes += 1
        
        print(f"  \u2713 Wall AprilTags (ID=0 pattern) on BACK WALL")
        print(f"    North: ({WALL_TAG_POSITIONS['north'][0]:.2f}, {WALL_TAG_POSITIONS['north'][1]:.2f}, {WALL_TAG_POSITIONS['north'][2]:.2f})")
        print(f"    South: ({WALL_TAG_POSITIONS['south'][0]:.2f}, {WALL_TAG_POSITIONS['south'][1]:.2f}, {WALL_TAG_POSITIONS['south'][2]:.2f})")
        print(f"    Size: {tag_size*100:.0f}cm x {tag_size*100:.0f}cm each")
        print(f"    Pattern: {total_black_cubes} black cells total")
        
        self.objects['wall_apriltags'] = wall_tags_path
    
    def _spawn_turtlebot(self):
        """Spawn TurtleBot3 Burger"""
        robot_cfg = self.config['robot_start']
        robot_pos = robot_cfg['position']
        
        turtlebot_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd"
        
        print(f"  Loading TurtleBot USD...")
        add_reference_to_stage(usd_path=turtlebot_usd, prim_path="/World/Turtlebot")
        
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Turtlebot",
                name="turtlebot",
                wheel_dof_names=["wheel_left_joint", "wheel_right_joint"],
                create_robot=False,
                position=np.array([robot_pos[0], robot_pos[1], 0.0]),
            )
        )
        
        print(f"  ✓ TurtleBot3 at [{robot_pos[0]}, {robot_pos[1]}]")
    
    def _create_robot_camera(self):
        """Create USD camera on robot facing FORWARD (+X direction)"""
        stage = self.world.stage
        
        self.camera_path = "/World/Turtlebot/base_link/robot_camera"
        camera_prim = UsdGeom.Camera.Define(stage, self.camera_path)
        
        camera_prim.CreateFocalLengthAttr(24.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateVerticalApertureAttr(15.2908)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.05, 0.0, 0.20))
        
        rotate_op = xformable.AddRotateZXYOp()
        rotate_op.Set(Gf.Vec3f(0, -90, -90))
        
        print(f"  ✓ Robot camera created at {self.camera_path} (facing +X)")


def setup_scene(tier='partial', seed=42, use_apriltag=True):
    """
    Setup scene - returns dict with world, robot, camera_path, objects.
    
    Args:
        tier: Scene tier ('partial' or 'heavy')
        seed: Random seed for reproducibility
        use_apriltag: If True, add AprilTag marker to charger (default: True)
    """
    config_path = f'config/scene_{tier}.yaml'
    scene = OccludeToDockScene(config_path=config_path)
    scene_dict = scene.setup(seed=seed, use_apriltag=use_apriltag)
    return scene_dict
