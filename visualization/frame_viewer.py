"""
Real-time Frame Viewer for Demo Visualization

This script watches the demo_realtime_frames folder and displays
the current.png file as it updates, giving you a live view of
the navigation without conflicting with Isaac Sim.

Usage:
    python visualization/frame_viewer.py

Requirements:
    pip install pillow

Controls:
    - Press 'q' or close window to quit
    - The window auto-updates every 100ms
"""

import os
import sys
import time

# Try to use tkinter for simple image display
try:
    import tkinter as tk
    from PIL import Image, ImageTk
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("tkinter or PIL not available - using fallback")


class FrameViewer:
    """Simple frame viewer using tkinter."""
    
    def __init__(self, watch_dir: str, update_interval_ms: int = 100):
        self.watch_dir = watch_dir
        self.update_interval = update_interval_ms
        self.current_path = os.path.join(watch_dir, "current.png")
        self.last_mtime = 0
        
        # Create window
        self.root = tk.Tk()
        self.root.title("Occlude-to-Dock: Live Visualization")
        self.root.configure(bg='black')
        
        # Create label for image
        self.label = tk.Label(self.root, bg='black')
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status = tk.Label(self.root, text="Waiting for frames...", 
                               bg='gray20', fg='white', anchor='w')
        self.status.pack(fill=tk.X)
        
        # Bind quit
        self.root.bind('q', lambda e: self.root.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        
        # Start update loop
        self.update_frame()
    
    def update_frame(self):
        """Check for new frame and update display."""
        try:
            if os.path.exists(self.current_path):
                mtime = os.path.getmtime(self.current_path)
                
                if mtime != self.last_mtime:
                    self.last_mtime = mtime
                    
                    # Load and display image
                    try:
                        img = Image.open(self.current_path)
                        
                        # Resize if too large
                        max_width = 1200
                        max_height = 800
                        if img.width > max_width or img.height > max_height:
                            ratio = min(max_width / img.width, max_height / img.height)
                            new_size = (int(img.width * ratio), int(img.height * ratio))
                            img = img.resize(new_size, Image.LANCZOS)
                        
                        photo = ImageTk.PhotoImage(img)
                        self.label.configure(image=photo)
                        self.label.image = photo  # Keep reference
                        
                        # Update status
                        self.status.configure(
                            text=f"Frame updated: {time.strftime('%H:%M:%S')} | Size: {img.width}x{img.height}"
                        )
                    except Exception as e:
                        self.status.configure(text=f"Error loading frame: {e}")
            else:
                self.status.configure(text=f"Waiting for {self.current_path}...")
                
        except Exception as e:
            self.status.configure(text=f"Error: {e}")
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_frame)
    
    def run(self):
        """Start the viewer."""
        self.root.mainloop()


def main():
    # Find the frames directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_dir = os.path.join(project_root, "demo_realtime_frames")
    
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        print(f"Created frames directory: {frames_dir}")
    
    print("="*60)
    print("  OCCLUDE-TO-DOCK: LIVE FRAME VIEWER")
    print("="*60)
    print(f"  Watching: {frames_dir}/current.png")
    print(f"  Press 'q' or close window to quit")
    print("="*60)
    
    if not HAS_TK:
        print("\nERROR: tkinter not available.")
        print("Install it or use an image viewer with auto-refresh:")
        print(f"  - Open {frames_dir}/current.png")
        print("  - Use IrfanView with Options > Properties > Auto-refresh")
        return
    
    viewer = FrameViewer(frames_dir)
    viewer.run()


if __name__ == "__main__":
    main()
