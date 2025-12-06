"""
AprilTag Texture Generator for Isaac Sim

Generates AprilTag images that can be used as textures on objects in Isaac Sim.

Usage:
    python generate_apriltag_texture.py

This will create:
    - assets/apriltags/charger_tag_0.png (Charger tag)
"""

import numpy as np
import os

# AprilTag 36h11 bit patterns for common IDs
# Each pattern is 10x10 (including 1-bit black border)
# Inner 6x6 is the data bits, surrounded by 1-bit black border, then 1-bit white border

# Tag 36h11 ID=0 pattern (the actual bit pattern)
TAG36H11_PATTERNS = {
    0: [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}


def create_apriltag_image(tag_id: int = 0, size: int = 512, border_ratio: float = 0.1) -> np.ndarray:
    """
    Create an AprilTag image with proper white border.
    """
    if tag_id in TAG36H11_PATTERNS:
        pattern = np.array(TAG36H11_PATTERNS[tag_id], dtype=np.uint8)
    else:
        print(f"⚠️ Tag ID {tag_id} pattern not stored, using ID 0")
        pattern = np.array(TAG36H11_PATTERNS[0], dtype=np.uint8)
    
    tag_size = int(size * (1 - 2 * border_ratio))
    scaled = np.kron(pattern, np.ones((tag_size // 10, tag_size // 10), dtype=np.uint8))
    
    img = np.ones((size, size), dtype=np.uint8) * 255
    border_px = int(size * border_ratio)
    img[border_px:border_px + scaled.shape[0], border_px:border_px + scaled.shape[1]] = scaled * 255
    
    img_rgb = np.stack([img, img, img], axis=-1)
    return img_rgb


def download_apriltag_from_github(tag_id: int = 0, output_dir: str = "assets") -> str:
    """Download AprilTag image from official repository."""
    import urllib.request
    
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/tag36_11_{tag_id:05d}.png"
    output_path = os.path.join(output_dir, f"tag36_11_{tag_id:05d}.png")
    
    try:
        print(f"Downloading AprilTag ID={tag_id} from GitHub...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def scale_apriltag_image(input_path: str, output_path: str, size: int = 512, add_border: bool = True):
    """Scale an AprilTag image and add white border."""
    try:
        import cv2
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if add_border:
            border = int(size * 0.15)
            tag_size = size - 2 * border
            img_scaled = cv2.resize(img, (tag_size, tag_size), interpolation=cv2.INTER_NEAREST)
            output = np.ones((size, size), dtype=np.uint8) * 255
            output[border:border+tag_size, border:border+tag_size] = img_scaled
            cv2.imwrite(output_path, output)
        else:
            img_scaled = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(output_path, img_scaled)
        
        print(f"✅ Scaled and saved to {output_path}")
    except ImportError:
        from PIL import Image
        img = Image.open(input_path)
        if add_border:
            border = int(size * 0.15)
            tag_size = size - 2 * border
            img_resized = img.resize((tag_size, tag_size), Image.NEAREST)
            output = Image.new('L', (size, size), 255)
            output.paste(img_resized, (border, border))
            output.save(output_path)
        else:
            img.resize((size, size), Image.NEAREST).save(output_path)


def generate_all_textures(output_dir: str = None):
    """Generate all required AprilTag textures for the simulation."""
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "assets", "apriltags")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Generating AprilTag Textures")
    print("="*60)
    
    charger_tag_path = os.path.join(output_dir, "charger_tag_0.png")
    downloaded = download_apriltag_from_github(tag_id=0, output_dir=output_dir)
    
    if downloaded:
        scale_apriltag_image(downloaded, charger_tag_path, size=512, add_border=True)
    else:
        print("Generating tag locally...")
        img = create_apriltag_image(tag_id=0, size=512)
        try:
            import cv2
            cv2.imwrite(charger_tag_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except ImportError:
            from PIL import Image
            Image.fromarray(img).save(charger_tag_path)
        print(f"✅ Generated {charger_tag_path}")
    
    print(f"\nTextures saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    generate_all_textures()
