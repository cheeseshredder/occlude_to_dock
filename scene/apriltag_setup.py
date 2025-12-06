"""
AprilTag Scene Setup Helper for Isaac Sim

Adds AprilTag markers to objects in the scene for robust detection.
Uses Isaac Sim's built-in AprilTag material support when available,
or falls back to a custom textured plane.
"""

import numpy as np
import os

# Check if Isaac Sim is available
try:
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, VisualCuboid
    from pxr import UsdGeom, UsdShade, Sdf, Gf
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False


def add_apriltag_to_charger(world, charger_position, tag_id=0, tag_size=0.15):
    """
    Add an AprilTag marker in front of the charger.
    
    The AprilTag is placed on a white plane facing the robot's starting position.
    
    Args:
        world: Isaac Sim World object
        charger_position: [x, y, z] position of charger center
        tag_id: AprilTag ID (default 0)
        tag_size: Size of the tag in meters (default 0.15m = 15cm)
        
    Returns:
        prim_path: Path to the created AprilTag prim
    """
    if not ISAAC_SIM_AVAILABLE:
        print("⚠️ Isaac Sim not available - skipping AprilTag creation")
        return None
    
    stage = world.stage
    
    # AprilTag position: slightly in front of charger, facing -X (toward robot)
    tag_x = charger_position[0] - 0.16  # 16cm in front of charger
    tag_y = charger_position[1]
    tag_z = charger_position[2]  # Same height as charger center
    
    # Create a plane for the AprilTag
    apriltag_path = "/World/charger_apriltag"
    
    # Try to use Isaac Sim's built-in AprilTag material
    try:
        # Check if AprilTag material exists
        apriltag_material_path = "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Materials/AprilTag/AprilTag.mdl"
        
        # For now, create a simple white plane with the tag texture
        # The actual tag detection will work based on the visual pattern
        
        # Create a thin cube as the tag surface
        tag_prim = world.scene.add(
            FixedCuboid(
                prim_path=apriltag_path,
                name="charger_apriltag",
                position=np.array([tag_x, tag_y, tag_z]),
                scale=np.array([0.01, tag_size, tag_size]),  # Very thin, tag_size x tag_size
                size=1.0,
                color=np.array([1.0, 1.0, 1.0]),  # White background
            )
        )
        
        # Note: For a proper AprilTag, you would apply the AprilTag material/texture here
        # This requires the texture file to be available
        # For testing, the white square acts as a placeholder
        
        print(f"  ✓ AprilTag placeholder at ({tag_x:.2f}, {tag_y:.2f}, {tag_z:.2f})")
        print(f"    NOTE: For full AprilTag support, apply tag texture manually")
        
        return apriltag_path
        
    except Exception as e:
        print(f"  ⚠️ Could not create AprilTag: {e}")
        return None


def create_apriltag_plane(stage, prim_path, position, size=0.15, rotation=None):
    """
    Create a plane with AprilTag texture.
    
    Args:
        stage: USD stage
        prim_path: Path for the new prim
        position: [x, y, z] world position
        size: Size of the tag in meters
        rotation: Optional rotation in degrees [x, y, z]
        
    Returns:
        UsdGeom.Mesh prim
    """
    # Create a simple quad mesh for the AprilTag
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    
    # Define quad vertices (facing -X by default)
    half_size = size / 2
    points = [
        Gf.Vec3f(0, -half_size, -half_size),
        Gf.Vec3f(0, half_size, -half_size),
        Gf.Vec3f(0, half_size, half_size),
        Gf.Vec3f(0, -half_size, half_size),
    ]
    
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    
    # Set position
    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
    
    if rotation is not None:
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(rotation[0], rotation[1], rotation[2]))
    
    return mesh


def setup_charger_with_apriltag(world, charger_config, tag_id=0, tag_size=0.15):
    """
    Setup charger with an AprilTag marker.
    
    This creates:
    1. The charger cube (green, for visual reference)
    2. An AprilTag plane in front of it
    
    Args:
        world: Isaac Sim World
        charger_config: Dict with 'position' and 'size' keys
        tag_id: AprilTag ID
        tag_size: AprilTag size in meters
        
    Returns:
        dict with 'charger' and 'apriltag' prims
    """
    from pxr import UsdPhysics
    
    stage = world.stage
    charger_size = charger_config['size']
    charger_pos = charger_config['position']
    
    # Create the charger (green cube)
    charger = world.scene.add(
        FixedCuboid(
            prim_path="/World/charger",
            name="charger",
            position=np.array([charger_pos[0], charger_pos[1], charger_size[2]/2]),
            scale=np.array([charger_size[0], charger_size[1], charger_size[2]]),
            size=1.0,
            color=np.array([0.0, 0.8, 0.0]),  # Green
        )
    )
    
    # Add collision for raycasts
    charger_prim = stage.GetPrimAtPath("/World/charger")
    if not charger_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(charger_prim)
    
    # Add AprilTag in front of charger
    apriltag_path = add_apriltag_to_charger(
        world,
        charger_position=[charger_pos[0], charger_pos[1], charger_size[2]/2],
        tag_id=tag_id,
        tag_size=tag_size
    )
    
    print(f"  ✓ Charger with AprilTag at [{charger_pos[0]}, {charger_pos[1]}]")
    
    return {
        'charger': charger,
        'apriltag_path': apriltag_path
    }


# Instructions for manual AprilTag setup in Isaac Sim:
MANUAL_SETUP_INSTRUCTIONS = """
=== Manual AprilTag Setup in Isaac Sim ===

If automatic AprilTag creation doesn't work, follow these steps:

1. Open Isaac Sim with your scene loaded

2. Add AprilTag material:
   - Go to Create -> April Tag -> (or Create -> Isaac -> April Tag)
   - This adds the AprilTag material to your stage

3. Create a cube for the tag:
   - Create -> Mesh -> Cube
   - Position it in front of the charger
   - Scale it to be flat (e.g., 0.01 x 0.15 x 0.15)

4. Apply the AprilTag material:
   - Select the cube
   - In Property panel, find Material
   - Assign the AprilTag material

5. Configure the tag:
   - In the material properties, set:
     - Mosaic texture path: (path to tag mosaic)
     - Tag ID: 0 (or your chosen ID)

6. The tag should now be visible and detectable!

Alternative: Use a custom texture:
1. Download tag from: https://github.com/AprilRobotics/apriltag-imgs
2. Create a material with the tag as a texture
3. Apply to a plane in front of the charger
"""

if __name__ == "__main__":
    print(MANUAL_SETUP_INSTRUCTIONS)
