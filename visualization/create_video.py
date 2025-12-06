"""
Convert saved visualization frames to video

Creates an MP4 video from the frame_XXXX.png files saved during
the demo run. Useful for presentations and documentation.

Usage:
    python visualization/create_video.py [--fps 10] [--output demo_video.mp4]

Requirements:
    pip install pillow imageio imageio-ffmpeg
"""

import os
import sys
import argparse
import glob

def create_video(frames_dir: str, output_path: str, fps: int = 10):
    """Create video from frame images."""
    
    try:
        import imageio
    except ImportError:
        print("ERROR: imageio not installed")
        print("Install with: pip install imageio imageio-ffmpeg")
        return False
    
    # Find all frames
    pattern = os.path.join(frames_dir, "frame_*.png")
    frame_files = sorted(glob.glob(pattern))
    
    if not frame_files:
        print(f"No frames found matching: {pattern}")
        print("Run the demo first: python test_pomdp_demo.py")
        return False
    
    print(f"Found {len(frame_files)} frames")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    
    # Create video
    print("Creating video...")
    
    try:
        with imageio.get_writer(output_path, fps=fps, codec='libx264', 
                                pixelformat='yuv420p', quality=8) as writer:
            for i, frame_path in enumerate(frame_files):
                if i % 10 == 0:
                    print(f"  Processing frame {i+1}/{len(frame_files)}...")
                
                img = imageio.imread(frame_path)
                writer.append_data(img)
        
        print(f"\n✅ Video saved to: {output_path}")
        print(f"   Duration: {len(frame_files) / fps:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"ERROR creating video: {e}")
        return False


def create_video_pillow(frames_dir: str, output_path: str, fps: int = 10):
    """Fallback: Create GIF using PIL (if imageio not available)."""
    
    from PIL import Image
    
    pattern = os.path.join(frames_dir, "frame_*.png")
    frame_files = sorted(glob.glob(pattern))
    
    if not frame_files:
        print(f"No frames found matching: {pattern}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    
    # Change output to GIF
    output_gif = output_path.replace('.mp4', '.gif')
    print(f"Output (GIF): {output_gif}")
    
    # Load frames
    print("Loading frames...")
    frames = []
    for i, frame_path in enumerate(frame_files):
        if i % 10 == 0:
            print(f"  Loading {i+1}/{len(frame_files)}...")
        img = Image.open(frame_path)
        # Reduce size for GIF
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        frames.append(img)
    
    # Save as GIF
    print("Saving GIF...")
    duration = int(1000 / fps)  # ms per frame
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"\n✅ GIF saved to: {output_gif}")
    print(f"   Duration: {len(frame_files) / fps:.1f} seconds")
    return True


def main():
    parser = argparse.ArgumentParser(description='Create video from demo frames')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--output', type=str, default='demo_video.mp4', 
                       help='Output video filename')
    parser.add_argument('--gif', action='store_true', 
                       help='Create GIF instead of MP4 (no ffmpeg needed)')
    args = parser.parse_args()
    
    # Find frames directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_dir = os.path.join(project_root, "demo_realtime_frames")
    output_path = os.path.join(project_root, args.output)
    
    print("="*60)
    print("  CREATE DEMO VIDEO")
    print("="*60)
    
    if args.gif:
        success = create_video_pillow(frames_dir, output_path, args.fps)
    else:
        try:
            import imageio
            success = create_video(frames_dir, output_path, args.fps)
        except ImportError:
            print("imageio not available, falling back to GIF...")
            success = create_video_pillow(frames_dir, output_path, args.fps)
    
    if success:
        print("\nVideo ready for presentation!")
    else:
        print("\nFailed to create video.")


if __name__ == "__main__":
    main()
