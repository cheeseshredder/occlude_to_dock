"""
Demo Recording Script - Heavy Occlusion with Belief Map + Detections

Creates frames showing:
- LEFT: Belief heatmap with robot position
- RIGHT: Camera view with bounding boxes and labels

Usage:
    isaac-python.bat run_demo.py
    
Then create video:
    ffmpeg -framerate 5 -i demo_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p demo_heavy.mp4
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Output directory
output_dir = os.path.join(project_root, "demo_frames")
os.makedirs(output_dir, exist_ok=True)

# Clean old frames
for f in os.listdir(output_dir):
    if f.endswith('.png'):
        os.remove(os.path.join(output_dir, f))

print("="*70)
print("DEMO RECORDING - HEAVY OCCLUSION")
print("="*70)

# Initialize Isaac Sim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

# Imports
from scene.setup_environment import setup_scene
from scene.robot_interface import IsaacSimRobot
from perception.apriltag_detector import HybridPerceptionModule, AprilTagChargerDetector
from perception.vision_model import PerceptionModule, create_isaac_raycast_func
from mapping.semantic_grid import SemanticGrid
from planner.qmdp_planner_real import QMDPPlannerReal
from language.parser import LanguageParser
from omni.isaac.core.utils.types import ArticulationAction
from omni.physx import get_physx_scene_query_interface

# Frame counter
frame_count = 0

def create_belief_heatmap(grid, robot_pose, size=400):
    """Create belief map visualization."""
    img = Image.new('RGB', (size, size), (30, 30, 35))
    draw = ImageDraw.Draw(img)
    
    cell_size = size // grid.width
    
    # Get data
    semantic_probs = grid.get_semantic_probs()
    occupancy = grid.get_occupancy_prob()
    CLASS_TO_ID = {'charger': 0, 'plant': 1, 'obstacle': 2, 'free': 3}
    
    # Draw cells
    for j in range(grid.height):
        for i in range(grid.width):
            y1 = (grid.height - 1 - j) * cell_size
            x1 = i * cell_size
            
            probs = semantic_probs[j, i]
            occ = occupancy[j, i]
            
            # Only show cells with significant occupancy (above uniform prior of 0.5)
            if occ > 0.55:
                charger_p = probs[CLASS_TO_ID['charger']]
                plant_p = probs[CLASS_TO_ID['plant']]
                
                # Only show charger (green) if probability is above uniform (0.25) AND dominant
                if charger_p > 0.35 and charger_p > plant_p:
                    intensity = min(255, int(80 + charger_p * 175))
                    color = (30, intensity, 30)
                # Only show plant (red) if probability is above uniform AND dominant
                elif plant_p > 0.35 and plant_p > charger_p:
                    intensity = min(255, int(80 + plant_p * 175))
                    color = (intensity, 40, 40)
                else:
                    # Generic occupied cell (gray)
                    color = (70, 70, 75)
                
                draw.rectangle([x1+1, y1+1, x1+cell_size-1, y1+cell_size-1], fill=color)
            
            # Grid lines
            draw.rectangle([x1, y1, x1+cell_size, y1+cell_size], outline=(50, 50, 55))
    
    # Draw locked estimates
    for class_name, color in [('charger', (0, 255, 0)), ('plant', (255, 80, 80))]:
        if grid.locked_estimates.get(class_name):
            pos = grid.locked_estimates[class_name]['position']
            i, j = grid.world_to_grid(pos[0], pos[1])
            px = i * cell_size + cell_size // 2
            py = (grid.height - 1 - j) * cell_size + cell_size // 2
            
            # X marker
            draw.line([(px-12, py-12), (px+12, py+12)], fill=color, width=3)
            draw.line([(px-12, py+12), (px+12, py-12)], fill=color, width=3)
            # Lock circle
            draw.ellipse([px-4, py-20, px+4, py-12], fill=color)
    
    # Draw robot
    rx, ry, ryaw = robot_pose
    i, j = grid.world_to_grid(rx, ry)
    px = i * cell_size + cell_size // 2
    py = (grid.height - 1 - j) * cell_size + cell_size // 2
    
    # Robot circle
    draw.ellipse([px-10, py-10, px+10, py+10], fill=(50, 100, 255), outline=(255, 255, 255), width=2)
    
    # Heading arrow
    ax = int(px + 25 * np.cos(ryaw))
    ay = int(py - 25 * np.sin(ryaw))
    draw.line([(px, py), (ax, ay)], fill=(255, 220, 0), width=3)
    
    return img


def create_camera_with_detections(rgb, detections, size=(480, 360)):
    """Create camera view with detection overlays."""
    img = Image.fromarray(rgb)
    img = img.resize(size, Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    
    scale_x = size[0] / 640
    scale_y = size[1] / 480
    
    colors = {'charger': (0, 255, 0), 'plant': (255, 80, 80), 'obstacle': (150, 150, 150)}
    
    for det in detections:
        if 'bbox_2d' not in det:
            continue
        
        x1, y1, x2, y2 = det['bbox_2d']
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        
        cls = det.get('class', 'unknown')
        color = colors.get(cls, (255, 255, 0))
        conf = det.get('confidence', 0)
        dist = det.get('estimated_distance', 0)
        
        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label
        label = f"{cls} {dist:.1f}m ({conf:.0%})"
        bbox = draw.textbbox((x1, y1-16), label)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1-16), label, fill=(0, 0, 0))
    
    return img


def create_combined_frame(belief_img, camera_img, action, step, elapsed, charger_conf, success=False):
    """Combine belief map and camera into one frame with status bar."""
    global frame_count
    
    # Canvas: belief (400) + camera (480) + padding
    width = 900
    height = 480
    
    canvas = Image.new('RGB', (width, height), (25, 25, 30))
    draw = ImageDraw.Draw(canvas)
    
    # Paste belief map (left)
    canvas.paste(belief_img, (10, 40))
    
    # Paste camera (right)
    canvas.paste(camera_img, (420, 40))
    
    # Title
    draw.text((10, 10), "Belief Map", fill=(255, 255, 255))
    draw.text((420, 10), "Camera + Detections", fill=(255, 255, 255))
    
    # Status bar at bottom
    bar_y = 410
    draw.rectangle([0, bar_y, width, height], fill=(40, 40, 45))
    
    # Status text
    status = f"Step: {step}  |  Action: {action}  |  Time: {elapsed:.1f}s"
    draw.text((15, bar_y + 10), status, fill=(200, 200, 200))
    
    # Charger confidence
    conf_text = f"Charger Confidence: {charger_conf:.0%}" if charger_conf else "Charger: Searching..."
    conf_color = (0, 255, 0) if charger_conf and charger_conf > 0.5 else (255, 200, 100)
    draw.text((15, bar_y + 35), conf_text, fill=conf_color)
    
    # Success indicator
    if success:
        draw.text((width - 150, bar_y + 20), "✓ DOCKED!", fill=(0, 255, 0))
    
    # Tier label
    draw.text((width - 150, bar_y + 45), "HEAVY OCCLUSION", fill=(255, 100, 100))
    
    # Save frame
    frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    canvas.save(frame_path)
    frame_count += 1
    
    return frame_path


# ============== MAIN DEMO ==============

print("\n[Demo] Setting up scene...")

# Setup scene - HEAVY occlusion
scene = setup_scene('heavy', seed=42, use_apriltag=True)
world = scene['world']
robot = scene['robot']
objects = scene['objects']
camera_path = scene['camera_path']

robot_interface = IsaacSimRobot(robot=robot, world=world, camera_path=camera_path, objects=objects)

# Parse instruction
language_parser = LanguageParser()
instruction = "dock at the charger behind the plant"
parsed = language_parser.parse(instruction)
detection_prompts = language_parser.get_detection_prompts(parsed)

print(f"[Demo] Instruction: {instruction}")

# Perception
actual_hfov_deg = 2 * np.degrees(np.arctan(20.955 / (2 * 24.0)))

grounding_dino = PerceptionModule(
    model_name="IDEA-Research/grounding-dino-tiny",
    text_prompts=detection_prompts,
    platt_a=1.509, platt_b=1.598,
    threshold=0.2,
    use_depth_sensor=True,
    hfov_deg=actual_hfov_deg,
    img_width=640, img_height=480,
)

apriltag_detector = AprilTagChargerDetector(
    charger_tag_id=0,
    focal_length_mm=24.0,
    horizontal_aperture_mm=20.955,
    img_width=640, img_height=480,
)

perception = HybridPerceptionModule(
    grounding_dino_detector=grounding_dino,
    apriltag_detector=apriltag_detector,
)

# Grid and planner
grid = SemanticGrid(
    size_meters=(6.0, 6.0),
    resolution=0.5,
    origin=(-3.0, -3.0),
    close_range_threshold=1.2,
    lock_threshold=1.0,
    min_detections_to_lock=2,
)

planner = QMDPPlannerReal(
    grid=grid,
    reward_dock=100.0,
    reward_step=-1.0,
    lambda_risk=0.5,
    lambda_ig=2.0,
    high_conf_threshold=0.7,
    medium_conf_threshold=0.4,
    room_bounds=(-2.7, -2.7, 2.7, 2.7),
)

# Motion controller (simplified)
class SimpleMotion:
    WHEEL_RADIUS = 0.033
    WHEEL_BASE = 0.16
    
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world
    
    def get_pose(self):
        pos, quat = self.robot.get_world_pose()
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return float(pos[0]), float(pos[1]), float(yaw)
    
    def set_velocity(self, v_lin, v_ang):
        v_left = (v_lin - v_ang * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        v_right = (v_lin + v_ang * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        self.robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([v_left, v_right])))
    
    def move_to(self, tx, ty, max_steps=400):
        for _ in range(max_steps):
            x, y, yaw = self.get_pose()
            dx, dy = tx - x, ty - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 0.12:
                self.set_velocity(0, 0)
                return True
            
            target_yaw = np.arctan2(dy, dx)
            yaw_err = np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw))
            
            if abs(yaw_err) > 0.3:
                self.set_velocity(0, np.clip(yaw_err * 1.2, -0.6, 0.6))
            else:
                self.set_velocity(min(0.08, dist * 0.5), np.clip(yaw_err * 0.8, -0.3, 0.3))
            
            self.world.step(render=True)
        
        self.set_velocity(0, 0)
        return False

motion = SimpleMotion(robot, world)

# Raycast setup
physx_query = get_physx_scene_query_interface()
raycast_func = create_isaac_raycast_func(physx_query)

CAMERA_OFFSET_X = 0.05
CAMERA_HEIGHT = 0.20

def get_camera_world_pos(pose):
    rx, ry, ryaw = pose
    cam_x = rx + CAMERA_OFFSET_X * np.cos(ryaw)
    cam_y = ry + CAMERA_OFFSET_X * np.sin(ryaw)
    return np.array([cam_x, cam_y, CAMERA_HEIGHT])

# Initialize
world.reset()
print("[Demo] Warming up...")
for _ in range(200):
    world.step(render=True)

robot_interface.refresh_viewport()

pose = robot_interface.get_pose()
cam_pos = get_camera_world_pos(pose)
perception.set_raycast_func(raycast_func, cam_pos)

# Charger position
CHARGER_X, CHARGER_Y = 2.8, 0.0

# ============== NAVIGATION LOOP ==============

print("[Demo] Starting navigation...")
start_time = time.time()
max_steps = 30
success = False

for step in range(max_steps):
    # Pause and capture
    motion.set_velocity(0, 0)
    for _ in range(30):
        world.step(render=True)
    
    # Get state
    pose = motion.get_pose()
    cam_pos = get_camera_world_pos(pose)
    perception.update_camera_pose(cam_pos, pose[2])
    
    rgb = robot_interface.get_rgb()
    detections = perception.detect(rgb, pose)
    grid.update_from_detections(detections, pose)
    
    # Get charger estimate
    charger_pos, charger_conf = grid.get_charger_estimate()
    
    # Check success
    if charger_pos:
        dist = np.sqrt((pose[0] - CHARGER_X)**2 + (pose[1] - CHARGER_Y)**2)
        if dist < 0.4:
            success = True
    
    # Get action
    target, action = planner.get_navigation_target(pose)
    
    # Create visualization frame
    elapsed = time.time() - start_time
    belief_img = create_belief_heatmap(grid, pose)
    camera_img = create_camera_with_detections(rgb, detections)
    create_combined_frame(belief_img, camera_img, action, step, elapsed, charger_conf, success)
    
    print(f"  Step {step}: action={action}, conf={(charger_conf if charger_conf else 0):.2f}, frames={frame_count}")
    
    if success:
        print(f"\n[Demo] ✅ SUCCESS! Distance: {dist:.2f}m")
        # Save a few more success frames
        for _ in range(10):
            world.step(render=True)
            create_combined_frame(belief_img, camera_img, "DOCKED", step, elapsed, charger_conf, True)
        break
    
    # Execute action
    if target:
        motion.move_to(target[0], target[1], max_steps=300)

elapsed = time.time() - start_time

print(f"\n[Demo] Finished!")
print(f"  Success: {success}")
print(f"  Time: {elapsed:.1f}s")
print(f"  Frames saved: {frame_count}")
print(f"\nTo create video:")
print(f"  ffmpeg -framerate 5 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p demo_heavy.mp4")

simulation_app.close()
