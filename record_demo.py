"""
Demo Recording Script for Presentation

This script runs a single navigation trial with full visualization capture,
generating frames that can be converted into a demo video.

Features captured:
- Belief map with robot position and heading
- Grounding DINO detections with bounding boxes
- AprilTag detections
- QMDP Q-values and action selection
- Status bar with confidence, action, timing

Usage:
    isaac-python.bat record_demo.py --tier heavy
    isaac-python.bat record_demo.py --tier partial
    
Then convert to video:
    python visualization/create_video.py --fps 10 --output demo_heavy.mp4

Author: Abdelrahman Awwad
"""

import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Parse arguments BEFORE importing Isaac Sim
parser = argparse.ArgumentParser(description='Record demo for presentation')
parser.add_argument('--tier', type=str, default='heavy', choices=['partial', 'heavy'],
                    help='Occlusion tier')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--output-dir', type=str, default='demo_recording',
                    help='Output directory for frames')
parser.add_argument('--headless', action='store_true', help='Run without GUI')
args = parser.parse_args()

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create output directory
output_dir = os.path.join(project_root, args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Clean old frames
for f in os.listdir(output_dir):
    if f.endswith('.png'):
        os.remove(os.path.join(output_dir, f))

print("="*70)
print("DEMO RECORDING FOR PRESENTATION")
print("="*70)
print(f"Tier: {args.tier}")
print(f"Seed: {args.seed}")
print(f"Output: {output_dir}")
print("="*70)

# Initialize Isaac Sim
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

# Import modules
from scene.setup_environment import setup_scene
from scene.robot_interface import IsaacSimRobot
from perception.apriltag_detector import HybridPerceptionModule
from mapping.semantic_grid import SemanticGrid
from planner.qmdp_planner_real import QMDPPlannerReal
from omni.isaac.core.utils.types import ArticulationAction


class DemoVisualizer:
    """Combined visualizer for demo recording."""
    
    def __init__(self, grid, output_dir, cell_size=50):
        self.grid = grid
        self.output_dir = output_dir
        self.cell_size = cell_size
        self.frame_count = 0
        self.start_time = time.time()
        
        # Dimensions
        self.heatmap_width = grid.width * cell_size
        self.heatmap_height = grid.height * cell_size
        self.camera_width = 480
        self.camera_height = 360
        
        # Total canvas
        self.total_width = self.heatmap_width + self.camera_width + 40
        self.total_height = max(self.heatmap_height, self.camera_height) + 120
        
        # Colors
        self.colors = {
            'charger': (0, 200, 0),
            'plant': (200, 50, 50),
            'obstacle': (128, 128, 128),
            'robot': (50, 50, 200),
            'heading': (255, 200, 0),
        }
        
        print(f"[DemoVisualizer] Output: {output_dir}")
        print(f"[DemoVisualizer] Frame size: {self.total_width}x{self.total_height}")
    
    def create_frame(
        self,
        robot_pose,
        rgb_frame,
        detections,
        action,
        q_values,
        info
    ):
        """Create a combined visualization frame."""
        
        # Create canvas
        canvas = Image.new('RGB', (self.total_width, self.total_height), (30, 30, 35))
        draw = ImageDraw.Draw(canvas)
        
        # === LEFT: Belief Map ===
        heatmap = self._create_heatmap(robot_pose)
        canvas.paste(heatmap, (15, 15))
        
        # Heatmap title
        draw.text((15, self.heatmap_height + 20), "Belief Map", fill=(255, 255, 255))
        
        # === RIGHT: Camera + Detections ===
        if rgb_frame is not None:
            camera = self._create_camera_view(rgb_frame, detections)
            cam_x = self.heatmap_width + 25
            canvas.paste(camera, (cam_x, 15))
            
            # Camera title
            draw.text((cam_x, self.camera_height + 20), "Camera + Detections", fill=(255, 255, 255))
        
        # === BOTTOM: Status Bar ===
        self._draw_status_bar(draw, action, q_values, info)
        
        # Save frame
        frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
        canvas.save(frame_path)
        
        # Also save as current.png for live viewing
        current_path = os.path.join(self.output_dir, "current.png")
        canvas.save(current_path)
        
        self.frame_count += 1
        return frame_path
    
    def _create_heatmap(self, robot_pose):
        """Create belief heatmap."""
        img = Image.new('RGB', (self.heatmap_width, self.heatmap_height), (40, 40, 45))
        draw = ImageDraw.Draw(img)
        
        # Get beliefs
        semantic_probs = self.grid.get_semantic_probs()
        occupancy = self.grid.get_occupancy_prob()
        CLASS_TO_ID = {'charger': 0, 'plant': 1, 'obstacle': 2, 'free': 3}
        
        # Draw cells
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                y1 = (self.grid.height - 1 - j) * self.cell_size
                y2 = y1 + self.cell_size
                x1 = i * self.cell_size
                x2 = x1 + self.cell_size
                
                probs = semantic_probs[j, i]
                occ = occupancy[j, i]
                
                # Cell color based on class
                if occ > 0.5:
                    charger_p = probs[CLASS_TO_ID['charger']]
                    plant_p = probs[CLASS_TO_ID['plant']]
                    obstacle_p = probs[CLASS_TO_ID['obstacle']]
                    
                    if charger_p > plant_p and charger_p > obstacle_p and charger_p > 0.25:
                        intensity = min(255, int(100 + charger_p * 155))
                        color = (30, intensity, 30)
                    elif plant_p > charger_p and plant_p > obstacle_p and plant_p > 0.25:
                        intensity = min(255, int(100 + plant_p * 155))
                        color = (intensity, 40, 40)
                    elif obstacle_p > 0.25:
                        color = (90, 90, 90)
                    else:
                        color = (60, 60, 65)
                    
                    draw.rectangle([x1+2, y1+2, x2-2, y2-2], fill=color)
                
                # Grid lines
                draw.rectangle([x1, y1, x2, y2], outline=(60, 60, 65))
        
        # Draw markers for locked estimates
        for class_name, color in [('charger', self.colors['charger']), 
                                   ('plant', self.colors['plant'])]:
            pos, conf = self.grid.get_class_estimate(class_name)
            if pos is not None and conf > 0.15:
                i, j = self.grid.world_to_grid(pos[0], pos[1])
                px = i * self.cell_size + self.cell_size // 2
                py = (self.grid.height - 1 - j) * self.cell_size + self.cell_size // 2
                
                is_locked = self.grid.locked_estimates.get(class_name) is not None
                size = 15 if is_locked else 10
                width = 4 if is_locked else 2
                
                # Draw X marker
                draw.line([(px-size, py-size), (px+size, py+size)], fill=color, width=width)
                draw.line([(px-size, py+size), (px+size, py-size)], fill=color, width=width)
                
                # Lock indicator (circle)
                if is_locked:
                    draw.ellipse([px-5, py-size-15, px+5, py-size-5], fill=color)
        
        # Draw robot
        rx, ry, ryaw = robot_pose
        i, j = self.grid.world_to_grid(rx, ry)
        px = i * self.cell_size + self.cell_size // 2
        py = (self.grid.height - 1 - j) * self.cell_size + self.cell_size // 2
        
        # Robot circle
        r = 12
        draw.ellipse([px-r, py-r, px+r, py+r], fill=self.colors['robot'], outline=(255, 255, 255))
        
        # Heading arrow
        arrow_len = 25
        ax = int(px + arrow_len * np.cos(ryaw))
        ay = int(py - arrow_len * np.sin(ryaw))
        draw.line([(px, py), (ax, ay)], fill=self.colors['heading'], width=4)
        
        # Arrowhead
        angle = np.arctan2(-(ay - py), ax - px)
        for offset in [2.5, -2.5]:
            hx = ax - 10 * np.cos(angle + offset)
            hy = ay + 10 * np.sin(angle + offset)
            draw.line([(ax, ay), (int(hx), int(hy))], fill=self.colors['heading'], width=3)
        
        return img
    
    def _create_camera_view(self, rgb_frame, detections):
        """Create camera view with detection overlays."""
        # Resize frame
        img = Image.fromarray(rgb_frame)
        img = img.resize((self.camera_width, self.camera_height), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        
        scale_x = self.camera_width / 640
        scale_y = self.camera_height / 480
        
        # Draw detections
        for det in detections:
            if 'bbox_2d' not in det:
                continue
            
            x1, y1, x2, y2 = det['bbox_2d']
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            
            cls = det.get('class', 'unknown')
            color = self.colors.get(cls, (255, 255, 0))
            
            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Label
            dist = det.get('estimated_distance', 0)
            conf = det.get('confidence', 0)
            label = f"{cls} {dist:.1f}m ({conf:.0%})"
            
            # Label background
            text_bbox = draw.textbbox((x1, y1-18), label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
            draw.text((x1, y1-18), label, fill=(0, 0, 0))
        
        return img
    
    def _draw_status_bar(self, draw, action, q_values, info):
        """Draw status bar at bottom."""
        bar_y = self.total_height - 100
        
        # Background
        draw.rectangle([0, bar_y, self.total_width, self.total_height], fill=(45, 45, 50))
        
        # Title
        draw.text((15, bar_y + 5), "QMDP Navigation Demo", fill=(255, 255, 255))
        
        # Action and step
        step = info.get('step', 0)
        elapsed = time.time() - self.start_time
        draw.text((15, bar_y + 28), f"Step: {step}  |  Action: {action}  |  Time: {elapsed:.1f}s", 
                  fill=(200, 200, 200))
        
        # Charger info
        charger_pos, charger_conf = self.grid.get_charger_estimate()
        if charger_pos:
            charger_text = f"Charger: ({charger_pos[0]:.2f}, {charger_pos[1]:.2f}) conf={charger_conf:.0%}"
            locked = " [LOCKED]" if self.grid.locked_estimates.get('charger') else ""
            color = (0, 255, 0) if self.grid.locked_estimates.get('charger') else (150, 255, 150)
        else:
            charger_text = "Charger: Not detected"
            locked = ""
            color = (150, 150, 150)
        draw.text((15, bar_y + 51), charger_text + locked, fill=color)
        
        # Q-values (right side)
        if q_values:
            qv_x = self.total_width - 300
            draw.text((qv_x, bar_y + 5), "Q-values:", fill=(255, 255, 255))
            
            sorted_q = sorted(q_values.items(), key=lambda x: -x[1])[:5]
            y_offset = 25
            for i, (a, q) in enumerate(sorted_q):
                marker = "→" if a == action else " "
                color = (255, 255, 0) if a == action else (180, 180, 180)
                draw.text((qv_x, bar_y + y_offset + i*15), f"{marker} {a}: {q:.1f}", fill=color)
        
        # Confidence bar
        bar_x = self.heatmap_width + 30
        bar_w = 180
        bar_h = 18
        
        draw.text((bar_x, bar_y + 70), "Charger Conf:", fill=(255, 255, 255))
        draw.rectangle([bar_x + 100, bar_y + 70, bar_x + 100 + bar_w, bar_y + 70 + bar_h], 
                       outline=(150, 150, 150))
        
        if charger_conf:
            fill_w = int(bar_w * min(1.0, charger_conf))
            bar_color = (0, 255, 0) if charger_conf > 0.6 else (255, 165, 0) if charger_conf > 0.3 else (255, 100, 100)
            draw.rectangle([bar_x + 100, bar_y + 70, bar_x + 100 + fill_w, bar_y + 70 + bar_h], fill=bar_color)


class SmoothMotionController:
    """Motion controller for demo."""
    
    WHEEL_RADIUS = 0.033
    WHEEL_BASE = 0.16
    MAX_LINEAR_VEL = 0.1
    MAX_ANGULAR_VEL = 0.6
    
    def __init__(self, robot, world):
        self.robot = robot
        self.world = world
    
    def get_pose(self):
        pos, quat = self.robot.get_world_pose()
        x, y = float(pos[0]), float(pos[1])
        
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        yaw = r.as_euler('xyz')[2]
        
        return (x, y, yaw)
    
    def _send_velocity(self, linear, angular):
        v_left = (linear - angular * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        v_right = (linear + angular * self.WHEEL_BASE / 2) / self.WHEEL_RADIUS
        
        self.robot.apply_action(ArticulationAction(joint_velocities=[v_left, v_right]))
        
        for _ in range(2):
            self.world.step(render=True)
    
    def move_to(self, target_x, target_y, max_steps=500):
        for step in range(max_steps):
            x, y, yaw = self.get_pose()
            
            dx = target_x - x
            dy = target_y - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 0.12:
                self._send_velocity(0, 0)
                return True
            
            target_yaw = np.arctan2(dy, dx)
            yaw_error = self._normalize_angle(target_yaw - yaw)
            
            if abs(yaw_error) > 0.3:
                angular = np.clip(yaw_error * 1.0, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
                self._send_velocity(0, angular)
            else:
                linear = min(self.MAX_LINEAR_VEL, dist * 0.5)
                angular = np.clip(yaw_error * 0.8, -0.3, 0.3)
                self._send_velocity(linear, angular)
        
        self._send_velocity(0, 0)
        return False
    
    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def backup(self, distance=0.2):
        x, y, yaw = self.get_pose()
        target_x = x - distance * np.cos(yaw)
        target_y = y - distance * np.sin(yaw)
        
        for _ in range(30):
            self._send_velocity(-0.05, 0)
        self._send_velocity(0, 0)


def is_path_blocked_by_plant(robot_pos, charger_pos, plant_pos, clearance=0.4):
    """Check if plant blocks path to charger."""
    rx, ry = robot_pos
    cx, cy = charger_pos
    px, py = plant_pos
    
    dx = cx - rx
    dy = cy - ry
    path_length = np.sqrt(dx**2 + dy**2)
    
    if path_length < 0.01:
        return False
    
    dx_norm = dx / path_length
    dy_norm = dy / path_length
    
    px_rel = px - rx
    py_rel = py - ry
    
    projection = px_rel * dx_norm + py_rel * dy_norm
    
    if projection < 0 or projection > path_length:
        return False
    
    perp_dist = abs(px_rel * dy_norm - py_rel * dx_norm)
    return perp_dist < clearance


def run_demo():
    """Run single demo trial with visualization recording."""
    
    # Scene parameters
    charger_x, charger_y = 2.8, 0.0
    plant_x, plant_y = 2.0, 0.0
    
    if args.tier == 'heavy':
        start_y = 2.0
    else:
        start_y = 0.7
    
    print(f"\n[Demo] Setting up scene: tier={args.tier}, seed={args.seed}")
    
    # Setup scene
    world, robot, scene_objects = setup_scene(
        charger_pos=(charger_x, charger_y),
        plant_pos=(plant_x, plant_y),
        start_pos=(0.0, start_y),
        start_yaw=0.0
    )
    
    # Initialize components
    robot_interface = IsaacSimRobot(world, robot)
    
    grid = SemanticGrid(
        size_meters=(6.0, 6.0),
        resolution=0.5,
        origin=(-3.0, -3.0),
    )
    
    perception = HybridPerceptionModule(
        robot_interface=robot_interface,
        world=world,
        charger_position=(charger_x, charger_y),
        device='cuda'
    )
    
    planner = QMDPPlannerReal(grid=grid)
    
    motion = SmoothMotionController(robot, world)
    
    visualizer = DemoVisualizer(grid, output_dir)
    
    # Warmup
    print("[Demo] Warming up simulation...")
    for _ in range(50):
        world.step(render=True)
    
    print("[Demo] Starting navigation...")
    start_time = time.time()
    max_steps = 25
    success = False
    
    for step in range(max_steps):
        pose = motion.get_pose()
        rx, ry, ryaw = pose
        
        # Get perception
        detections, rgb_frame = perception.get_detections_with_frame(pose)
        
        # Filter detections
        valid_detections = [d for d in detections if d.get('estimated_distance', 999) < 5.0]
        
        # Update grid
        if valid_detections:
            grid.update_from_detections(valid_detections, pose)
        
        # Get charger info
        charger_pos, charger_conf = grid.get_charger_estimate()
        
        # Check for docking condition
        if charger_pos:
            dist_to_charger = np.sqrt((rx - charger_pos[0])**2 + (ry - charger_pos[1])**2)
            
            if dist_to_charger < 0.35:
                success = True
                print(f"  ✅ DOCKED! Distance: {dist_to_charger:.2f}m")
                
                # Record final frame
                visualizer.create_frame(
                    robot_pose=pose,
                    rgb_frame=rgb_frame,
                    detections=valid_detections,
                    action="DOCK_SUCCESS",
                    q_values={},
                    info={'step': step}
                )
                break
        
        # Get QMDP action
        target, action = planner.get_navigation_target(pose)
        
        # Get Q-values for visualization
        belief = grid.get_charger_belief()
        q_values = {}
        for a in ["DOCK", "APPROACH", "EXPLORE", "RESCAN", "ROTATE"]:
            q_values[a] = planner._compute_action_q_value(a, pose, belief)
        
        # Check path blocking
        plant_pos_est, _ = grid.get_class_estimate('plant')
        if charger_pos and plant_pos_est:
            if is_path_blocked_by_plant((rx, ry), charger_pos, plant_pos_est):
                if action == "APPROACH":
                    action = "APPROACH_AROUND"
                    print(f"  🔄 Path blocked - APPROACH_AROUND")
        
        # Record frame
        visualizer.create_frame(
            robot_pose=pose,
            rgb_frame=rgb_frame,
            detections=valid_detections,
            action=action,
            q_values=q_values,
            info={'step': step}
        )
        
        # Execute action
        if target:
            motion.move_to(target[0], target[1], max_steps=300)
        
        print(f"  Step {step}: action={action}, charger_conf={charger_conf:.2f if charger_conf else 0:.2f}")
    
    elapsed = time.time() - start_time
    
    print(f"\n[Demo] Finished!")
    print(f"  Success: {success}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Frames: {visualizer.frame_count}")
    print(f"\nTo create video:")
    print(f"  python visualization/create_video.py --fps 5 --output demo_{args.tier}.mp4")
    
    simulation_app.close()


if __name__ == "__main__":
    run_demo()
