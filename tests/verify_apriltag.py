"""
Verify AprilTag Pattern and Test Detection

This script:
1. Downloads the official tag36h11 ID 0 from AprilRobotics
2. Compares it with our pattern
3. Tests detection on both

Run with: python verify_apriltag.py
"""

import numpy as np
import cv2
import os
import urllib.request

# Our current pattern (from setup_environment.py)
OUR_PATTERN = np.array([
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
], dtype=np.uint8)

def download_official_tag():
    """Download official tag36h11 ID 0 from AprilRobotics."""
    url = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/tag36_11_00000.png"
    
    try:
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, "official_tag36h11_id0.png")
        img = cv2.imread("official_tag36h11_id0.png", cv2.IMREAD_GRAYSCALE)
        print(f"Downloaded: shape={img.shape}")
        return img
    except Exception as e:
        print(f"Failed to download: {e}")
        return None

def extract_pattern_from_image(img):
    """Extract 10x10 binary pattern from AprilTag image."""
    h, w = img.shape
    cell_h = h // 10
    cell_w = w // 10
    
    pattern = np.zeros((10, 10), dtype=np.uint8)
    
    for row in range(10):
        for col in range(10):
            # Sample center of each cell
            cy = row * cell_h + cell_h // 2
            cx = col * cell_w + cell_w // 2
            pixel = img[cy, cx]
            pattern[row, col] = 1 if pixel > 127 else 0
    
    return pattern

def create_test_image(pattern, size=200):
    """Create a test image from pattern."""
    cell_size = size // 10
    img = np.zeros((size, size), dtype=np.uint8)
    
    for row in range(10):
        for col in range(10):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            if pattern[row, col] == 1:
                img[y1:y2, x1:x2] = 255
    
    return img

def test_detection(img, label):
    """Test AprilTag detection on an image."""
    from pupil_apriltags import Detector
    
    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    detections = detector.detect(img)
    
    print(f"\n{label}:")
    print(f"  Image size: {img.shape}")
    if len(detections) == 0:
        print(f"  ❌ No detections")
    else:
        for det in detections:
            print(f"  ✅ Detected ID: {det.tag_id}")
            print(f"     Decision margin: {det.decision_margin:.1f}")
    
    return detections

def main():
    print("="*60)
    print("AprilTag Pattern Verification")
    print("="*60)
    
    # 1. Download official tag
    print("\n[1] Downloading official tag36h11 ID 0...")
    official_img = download_official_tag()
    
    if official_img is not None:
        # Extract pattern
        official_pattern = extract_pattern_from_image(official_img)
        
        print("\n[2] Comparing patterns...")
        print("\nOfficial pattern:")
        print(official_pattern)
        
        print("\nOur pattern:")
        print(OUR_PATTERN)
        
        # Compare
        match = np.array_equal(official_pattern, OUR_PATTERN)
        print(f"\nPatterns match: {'✅ YES' if match else '❌ NO'}")
        
        if not match:
            print("\nDifferences (1 = different):")
            diff = (official_pattern != OUR_PATTERN).astype(int)
            print(diff)
            print(f"\nTotal different cells: {np.sum(diff)}")
            
            # Show what the correct pattern should be
            print("\n" + "="*60)
            print("CORRECT PATTERN FOR setup_environment.py:")
            print("="*60)
            print("\nAPRILTAG_36H11_ID0 = np.array([")
            for row in range(10):
                row_str = "    [" + ", ".join(str(x) for x in official_pattern[row]) + "],"
                print(row_str)
            print("], dtype=np.uint8)")
    
    # 3. Test detection on generated images
    print("\n[3] Testing detection...")
    
    # Test on official image (scaled up)
    if official_img is not None:
        official_big = cv2.resize(official_img, (200, 200), interpolation=cv2.INTER_NEAREST)
        test_detection(official_big, "Official tag (resized to 200x200)")
    
    # Test on our pattern
    our_img = create_test_image(OUR_PATTERN, 200)
    test_detection(our_img, "Our pattern (generated 200x200)")
    
    # If we have the correct pattern, test that too
    if official_img is not None and not match:
        correct_img = create_test_image(official_pattern, 200)
        test_detection(correct_img, "Correct pattern (generated 200x200)")
    
    # 4. Test on captured image if it exists
    captured_path = "outputs/apriltag_test/camera_capture.png"
    if os.path.exists(captured_path):
        print(f"\n[4] Testing on captured image: {captured_path}")
        captured = cv2.imread(captured_path, cv2.IMREAD_GRAYSCALE)
        test_detection(captured, "Captured from Isaac Sim")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()
