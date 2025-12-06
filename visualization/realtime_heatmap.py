"""
Real-time Belief Heatmap Visualization for Demo/Presentation

This version saves frames to disk for viewing, avoiding GUI conflicts with Isaac Sim.
A separate viewer window can be used to watch the frames update.

Usage:
    from visualization.realtime_heatmap import RealtimeVisualizer
    
    viz = RealtimeVisualizer(grid, output_dir="demo_frames")
    
    # In navigation loop:
    viz.update(robot_pose, detections, rgb_frame, action_name, info_dict)
    
    # At end:
    viz.close()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from PIL import Image, ImageDraw, ImageFont


class RealtimeVisualizer:
    """
    Real-time visualization that saves frames to disk.
    
    Creates combined images showing:
    - Left: Belief heatmap with robot/charger/plant markers
    - Right: Camera view with detection overlays
    - Bottom: Status bar with action, confidence, timing
    
    Frames are saved to output_dir and can be viewed with any image viewer.
    The 'current.png' file is continuously overwritten for "live" viewing.
    """
    
    def __init__(
        self,
        grid,
        window_name: str = "Occlude-to-Dock Demo",
        show_camera: bool = True,
        output_dir: str = "demo_realtime_frames",
        cell_size: int = 40,
        camera_size: Tuple[int, int] = (400, 300),
        save_all_frames: bool = False,
    ):
        """
        Initialize the visualizer.
        
        Args:
            grid: SemanticGrid instance
            window_name: Title for the visualization
            show_camera: Whether to include camera feed
            output_dir: Directory to save frames
            cell_size: Size of each grid cell in pixels
            camera_size: (width, height) for camera display
            save_all_frames: If True, save numbered frames; if False, only save current.png
        """
        self.grid = grid
        self.window_name = window_name
        self.show_camera = show_camera
        self.output_dir = output_dir
        self.cell_size = cell_size
        self.camera_size = camera_size
        self.save_all_frames = save_all_frames
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean old frames
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        
        # Calculate dimensions
        self.heatmap_width = grid.width * cell_size
        self.heatmap_height = grid.height * cell_size
        
        if show_camera:
            self.total_width = self.heatmap_width + camera_size[0] + 30
        else:
            self.total_width = self.heatmap_width + 20
        
        self.total_height = max(self.heatmap_height, camera_size[1]) + 100  # Status bar
        
        # Timing
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Colors (RGB for PIL)
        self.colors = {
            'charger': (0, 255, 0),      # Green
            'plant': (255, 0, 0),         # Red
            'obstacle': (128, 128, 128),  # Gray
            'robot': (0, 0, 255),         # Blue
            'robot_dir': (255, 255, 0),   # Yellow
        }
        
        self._is_open = True
        self._frame_number = 0
        
        print(f"[RealtimeVisualizer] Saving frames to: {output_dir}/")
        print(f"  Grid: {grid.width}×{grid.height} cells")
        print(f"  Frame size: {self.total_width}×{self.total_height}")
        print(f"  View 'current.png' in the output folder for live updates")
    
    def _create_heatmap(self, robot_pose: Tuple[float, float, float]) -> Image.Image:
        """Create the belief heatmap image."""
        img = Image.new('RGB', (self.heatmap_width, self.heatmap_height), (220, 220, 220))
        draw = ImageDraw.Draw(img)
        
        # Get semantic probabilities and occupancy (like original visualize method)
        semantic_probs = self.grid.get_semantic_probs()
        occupancy = self.grid.get_occupancy_prob()
        
        # Class IDs
        CLASS_TO_ID = {'charger': 0, 'plant': 1, 'obstacle': 2, 'free': 3}
        
        # Draw cells with semantic coloring (matching original visualize method)
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                # Flip j for display
                y1 = (self.grid.height - 1 - j) * self.cell_size
                y2 = y1 + self.cell_size
                x1 = i * self.cell_size
                x2 = x1 + self.cell_size
                
                probs = semantic_probs[j, i]
                occ = occupancy[j, i]
                
                # Only color if occupied
                if occ > 0.55:
                    charger_p = probs[CLASS_TO_ID['charger']]
                    plant_p = probs[CLASS_TO_ID['plant']]
                    obstacle_p = probs[CLASS_TO_ID['obstacle']]
                    
                    if charger_p > plant_p and charger_p > obstacle_p and charger_p > 0.3:
                        # Green for charger
                        intensity = min(255, int(150 + charger_p * 105))
                        color = (50, intensity, 50)
                    elif plant_p > charger_p and plant_p > obstacle_p and plant_p > 0.3:
                        # Red for plant
                        intensity = min(255, int(150 + plant_p * 105))
                        color = (intensity, 50, 50)
                    elif obstacle_p > 0.3:
                        # Gray for obstacle
                        color = (100, 100, 100)
                    else:
                        # Light gray for occupied but unknown
                        color = (180, 180, 180)
                    
                    draw.rectangle([x1+2, y1+2, x2-2, y2-2], fill=color)
                
                # Grid lines
                draw.rectangle([x1, y1, x2, y2], outline=(180, 180, 180))
        
        # Draw class markers
        self._draw_markers(draw)
        
        # Draw robot
        self._draw_robot(draw, robot_pose)
        
        return img
    
    def _draw_markers(self, draw: ImageDraw.Draw):
        """Draw X markers for detected classes."""
        markers = {
            'charger': self.colors['charger'],
            'plant': self.colors['plant'],
            'obstacle': self.colors['obstacle'],
        }
        
        for class_name, color in markers.items():
            pos, conf = self.grid.get_class_estimate(class_name)
            if pos is not None and conf > 0.1:
                i, j = self.grid.world_to_grid(pos[0], pos[1])
                px = i * self.cell_size + self.cell_size // 2
                py = (self.grid.height - 1 - j) * self.cell_size + self.cell_size // 2
                
                is_locked = self.grid.locked_estimates.get(class_name) is not None
                width = 3 if is_locked else 2
                size = 12
                
                # Draw X
                draw.line([(px-size, py-size), (px+size, py+size)], fill=color, width=width)
                draw.line([(px-size, py+size), (px+size, py-size)], fill=color, width=width)
                
                # Lock indicator
                if is_locked:
                    draw.ellipse([px-4, py-size-12, px+4, py-size-4], fill=color)
    
    def _draw_robot(self, draw: ImageDraw.Draw, pose: Tuple[float, float, float]):
        """Draw robot position and heading."""
        rx, ry, ryaw = pose
        
        i, j = self.grid.world_to_grid(rx, ry)
        px = i * self.cell_size + self.cell_size // 2
        py = (self.grid.height - 1 - j) * self.cell_size + self.cell_size // 2
        
        # Robot circle
        r = 10
        draw.ellipse([px-r, py-r, px+r, py+r], fill=self.colors['robot'], outline=(0, 0, 0))
        
        # Heading arrow
        arrow_len = 20
        ax = int(px + arrow_len * np.cos(ryaw))
        ay = int(py - arrow_len * np.sin(ryaw))  # Flip Y
        
        draw.line([(px, py), (ax, ay)], fill=self.colors['robot_dir'], width=3)
        
        # Arrowhead
        angle = np.arctan2(-(ay - py), ax - px)
        head_len = 8
        for offset in [2.5, -2.5]:
            hx = ax - head_len * np.cos(angle + offset)
            hy = ay + head_len * np.sin(angle + offset)
            draw.line([(ax, ay), (int(hx), int(hy))], fill=self.colors['robot_dir'], width=2)
    
    def _create_camera_overlay(self, rgb_frame: np.ndarray, detections: List[Dict]) -> Image.Image:
        """Create camera view with detection overlays."""
        img = Image.fromarray(rgb_frame)
        img = img.resize(self.camera_size, Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        
        scale_x = self.camera_size[0] / 640
        scale_y = self.camera_size[1] / 480
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox_2d']
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            
            color = self.colors.get(det['class'], (255, 255, 0))
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            dist_str = f"{det['estimated_distance']:.1f}m"
            method = "A" if det.get('depth_method') == 'apriltag_pose' else \
                     "R" if det.get('depth_method') == 'raycast' else "B"
            label = f"{det['class']} {dist_str} [{method}]"
            
            draw.rectangle([x1, max(0, y1-14), x1+len(label)*6+4, y1], fill=color)
            draw.text((x1+2, max(0, y1-13)), label, fill=(0, 0, 0))
        
        return img
    
    def _create_status_bar(self, action: str, info: Dict, width: int) -> Image.Image:
        """Create status bar."""
        bar_height = 90
        img = Image.new('RGB', (width, bar_height), (50, 50, 50))
        draw = ImageDraw.Draw(img)
        
        # Get estimates
        charger_pos, charger_conf = self.grid.get_charger_estimate()
        plant_pos, plant_conf = self.grid.get_class_estimate('plant')
        
        charger_locked = self.grid.locked_estimates.get('charger') is not None
        plant_locked = self.grid.locked_estimates.get('plant') is not None
        
        elapsed = time.time() - self.start_time
        step = info.get('step', 0)
        
        # FPS
        self.frame_count += 1
        if time.time() - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (time.time() - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = time.time()
        
        # Text
        y = 8
        draw.text((10, y), f"Step: {step} | Action: {action} | Time: {elapsed:.1f}s | FPS: {self.fps:.1f}", 
                 fill=(255, 255, 255))
        
        y += 22
        if charger_pos:
            lock_str = "[LOCKED]" if charger_locked else ""
            text = f"Charger: ({charger_pos[0]:.2f}, {charger_pos[1]:.2f}) conf={charger_conf:.2f} {lock_str}"
            color = (0, 255, 0) if charger_locked else (0, 200, 0)
        else:
            text = "Charger: Not detected"
            color = (150, 150, 150)
        draw.text((10, y), text, fill=color)
        
        y += 22
        if plant_pos:
            lock_str = "[LOCKED]" if plant_locked else ""
            text = f"Plant: ({plant_pos[0]:.2f}, {plant_pos[1]:.2f}) {lock_str}"
            color = (255, 100, 100) if plant_locked else (200, 100, 100)
        else:
            text = "Plant: Not detected"
            color = (150, 150, 150)
        draw.text((10, y), text, fill=color)
        
        # Confidence bar
        bar_x = width - 220
        bar_y = 30
        bar_w = 150
        bar_h = 20
        
        draw.text((bar_x - 50, bar_y + 2), "Conf:", fill=(255, 255, 255))
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(150, 150, 150))
        
        fill_w = int(bar_w * min(1.0, charger_conf))
        if fill_w > 0:
            bar_color = (0, 255, 0) if charger_conf > 0.6 else (255, 165, 0)
            draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=bar_color)
        
        return img
    
    def update(
        self,
        robot_pose: Tuple[float, float, float],
        detections: List[Dict],
        rgb_frame: Optional[np.ndarray] = None,
        action: str = "",
        info: Optional[Dict] = None,
    ) -> bool:
        """
        Update the visualization and save frame.
        
        Returns:
            True (always continues - no window to close)
        """
        if not self._is_open:
            return False
        
        if info is None:
            info = {}
        
        try:
            # Create heatmap
            heatmap = self._create_heatmap(robot_pose)
            
            # Create combined image
            content_height = max(self.heatmap_height, self.camera_size[1])
            combined = Image.new('RGB', (self.total_width, self.total_height), (240, 240, 240))
            
            # Place heatmap
            combined.paste(heatmap, (10, 10))
            
            # Place camera
            if self.show_camera and rgb_frame is not None:
                camera = self._create_camera_overlay(rgb_frame, detections)
                cam_x = self.heatmap_width + 20
                cam_y = (content_height - self.camera_size[1]) // 2 + 10
                combined.paste(camera, (cam_x, cam_y))
            
            # Create and place status bar
            status = self._create_status_bar(action, info, self.total_width)
            combined.paste(status, (0, content_height + 10))
            
            # Add title
            draw = ImageDraw.Draw(combined)
            # Legend at bottom of heatmap
            legend_y = self.heatmap_height + 15
            draw.rectangle([10, legend_y, self.heatmap_width, legend_y + 25], fill=(60, 60, 60))
            
            x = 20
            for name, color in [('Charger', (0, 255, 0)), ('Plant', (255, 0, 0)), 
                               ('Robot', (0, 0, 255)), ('Heading', (255, 255, 0))]:
                draw.rectangle([x, legend_y + 5, x + 15, legend_y + 20], fill=color)
                draw.text((x + 20, legend_y + 5), name, fill=(255, 255, 255))
                x += 80
            
            # Save frame
            current_path = os.path.join(self.output_dir, "current.png")
            combined.save(current_path)
            
            if self.save_all_frames:
                frame_path = os.path.join(self.output_dir, f"frame_{self._frame_number:04d}.png")
                combined.save(frame_path)
            
            self._frame_number += 1
            
            return True
            
        except Exception as e:
            print(f"[RealtimeVisualizer] Error: {e}")
            import traceback
            traceback.print_exc()
            return True  # Continue anyway
    
    def close(self):
        """Mark visualizer as closed."""
        self._is_open = False
        print(f"[RealtimeVisualizer] Closed. Frames saved to: {self.output_dir}/")
        print(f"  Total frames: {self._frame_number}")
    
    def save_frame(self, filename: str):
        """Save current frame with custom filename."""
        # This would need the last frame stored - for now just print
        print(f"[RealtimeVisualizer] Use output_dir for frames")


# Standalone test
if __name__ == "__main__":
    import sys
    
    print("Real-time Heatmap Visualizer - Test Mode (File-based)")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from mapping.semantic_grid import SemanticGrid
    
    # Create grid
    grid = SemanticGrid(
        size_meters=(6.0, 6.0),
        resolution=0.5,
        origin=(-3.0, -3.0),
    )
    
    # Create visualizer
    viz = RealtimeVisualizer(grid, show_camera=False, save_all_frames=True)
    
    # Simulate some data
    robot_pose = (0.0, 0.0, 0.0)
    
    # Add fake detections
    fake_detections = [
        {
            'class': 'plant',
            'world_position': (2.0, 0.0),
            'confidence': 0.8,
            'calibrated_prob': 0.75,
            'color_score': 0.9,
            'estimated_distance': 2.0,
            'depth_method': 'raycast',
        },
        {
            'class': 'charger',
            'world_position': (2.8, 0.0),
            'confidence': 0.6,
            'calibrated_prob': 0.55,
            'color_score': 0.7,
            'estimated_distance': 2.8,
            'depth_method': 'bbox',
        },
    ]
    
    grid.update_from_detections(fake_detections, robot_pose)
    
    print(f"Generating 50 test frames...")
    print(f"View: {viz.output_dir}/current.png")
    
    for step in range(50):
        t = step * 0.1
        robot_pose = (t * 0.5, np.sin(t) * 0.3, t * 0.2)
        
        viz.update(
            robot_pose=robot_pose,
            detections=[],
            rgb_frame=None,
            action="APPROACH",
            info={'step': step}
        )
        
        time.sleep(0.05)
    
    viz.close()
    print("Done! Check the output folder for frames.")
