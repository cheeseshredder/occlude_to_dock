"""
Convert demo frames to MP4 video using OpenCV.
No ffmpeg required!

Usage:
    pip install opencv-python
    python make_video.py
"""

import cv2
import os

frames_dir = r"C:\Users\abdul\Desktop\occlude_to_dock\demo_frames"
output = r"C:\Users\abdul\Desktop\occlude_to_dock\demo_heavy.mp4"
fps = 5

# Get sorted frame list
frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

if not frames:
    print("No frames found in", frames_dir)
    exit(1)

print(f"Found {len(frames)} frames")

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
height, width = first_frame.shape[:2]
print(f"Frame size: {width}x{height}")

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, fps, (width, height))

# Write all frames
for i, f in enumerate(frames):
    frame = cv2.imread(os.path.join(frames_dir, f))
    out.write(frame)
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(frames)} frames")

out.release()
print(f"\n✅ Video saved to: {output}")
print(f"   Duration: {len(frames) / fps:.1f} seconds")
