"""
Camera capture for Isaac Sim - Uses viewport capture from robot camera
Bypasses broken Replicator/Camera sensor
"""

import numpy as np
import os


def get_camera_data(camera_path=None, world=None):
    """
    Capture RGB from Isaac Sim viewport (switched to robot camera).
    
    Args:
        camera_path: Path to camera prim (e.g., "/World/Turtlebot/base_link/robot_camera")
        world: World instance for stepping
        
    Returns:
        rgb: (480, 640, 3) uint8
        depth: (480, 640) float32 (placeholder - depth not available from viewport)
    """
    # Step world to render
    if world is not None:
        for _ in range(3):
            world.step(render=True)
    
    try:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_buffer
        
        viewport = get_active_viewport()
        
        # Switch to robot camera if path provided
        if camera_path is not None:
            viewport.set_active_camera(camera_path)
            # Step to render with new camera
            if world is not None:
                for _ in range(3):
                    world.step(render=True)
        
        # Try to capture to buffer
        try:
            buffer = capture_viewport_to_buffer(viewport)
            if buffer is not None:
                rgb = np.array(buffer, dtype=np.uint8)
                if len(rgb.shape) == 3 and rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]  # RGBA -> RGB
                
                # Resize if needed
                if rgb.shape[:2] != (480, 640):
                    from PIL import Image
                    img = Image.fromarray(rgb)
                    img = img.resize((640, 480), Image.LANCZOS)
                    rgb = np.array(img)
                
                return rgb, np.zeros((480, 640), dtype=np.float32)
        except:
            pass
        
        # Fallback: capture to temp file
        import tempfile
        from omni.kit.viewport.utility import capture_viewport_to_file
        from PIL import Image
        
        temp_path = os.path.join(tempfile.gettempdir(), "isaac_capture_temp.png")
        capture_viewport_to_file(viewport, temp_path)
        
        if os.path.exists(temp_path):
            img = Image.open(temp_path)
            img = img.resize((640, 480), Image.LANCZOS)
            rgb = np.array(img)
            if len(rgb.shape) == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            os.remove(temp_path)
            return rgb.astype(np.uint8), np.zeros((480, 640), dtype=np.float32)
        
    except Exception as e:
        print(f"⚠ Camera capture error: {e}")
    
    # Return black image if all fails
    return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.float32)


def get_ground_truth(world, objects, camera_path=None):
    """Extract ground truth from scene objects."""
    gt = {}
    
    try:
        if 'charger' in objects and objects['charger'] is not None:
            charger = objects['charger']
            charger_pos, _ = charger.get_world_pose()
            gt['charger_visible'] = True
            gt['charger_bbox'] = None
            gt['charger_pose'] = charger_pos.tolist() if hasattr(charger_pos, 'tolist') else list(charger_pos)
        
        if 'plant' in objects and objects['plant'] is not None:
            plant = objects['plant']
            plant_pos, _ = plant.get_world_pose()
            gt['plant_visible'] = True
            gt['plant_bbox'] = None
            gt['plant_pose'] = plant_pos.tolist() if hasattr(plant_pos, 'tolist') else list(plant_pos)
        
        gt['distractors'] = []
        for key in ['distractor_1', 'distractor_2']:
            if key in objects and objects[key] is not None:
                dist = objects[key]
                dist_pos, _ = dist.get_world_pose()
                gt['distractors'].append({
                    'name': key,
                    'visible': True,
                    'bbox': None,
                    'pose': dist_pos.tolist() if hasattr(dist_pos, 'tolist') else list(dist_pos)
                })
    except Exception as e:
        print(f"⚠ Ground truth warning: {e}")
    
    return gt
