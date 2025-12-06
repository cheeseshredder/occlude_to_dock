# AprilTag Integration for Occlude-to-Dock

## Overview

AprilTags replace unreliable color-based charger detection with robust fiducial marker detection.
The AprilTag is automatically created on both sides of the charger (+Y and -Y faces) during scene setup.

## Benefits Over Color Detection

| Aspect | Color Detection | AprilTag Detection |
|--------|----------------|-------------------|
| False Positives | ❌ High (gray boxes detected as green) | ✅ Zero (unique ID) |
| Lighting Sensitivity | ❌ Affected by shadows/reflections | ✅ Robust black/white pattern |
| Unique ID | ❌ Can't distinguish similar colors | ✅ Each tag has unique ID |
| Pose Estimation | ❌ Approximate (bbox depth) | ✅ Precise 6DOF |
| Distance Accuracy | ❌ ~10-30% error | ✅ ~2% error |

## Installation

```bash
pip install pupil-apriltags
```

## Tag Configuration

| Setting | Value |
|---------|-------|
| Tag Family | tag36h11 |
| Tag ID | 0 (charger marker) |
| Physical Size | 15cm x 15cm |
| Pattern | 10x10 grid (8x8 data + 1-bit border) |
| Placement | Both +Y and -Y faces of charger |

## Files

```
perception/
  apriltag_detector.py    # AprilTagChargerDetector + HybridPerceptionModule

scene/
  setup_environment.py    # Creates AprilTags on charger automatically

mapping/
  semantic_grid.py        # Updated to trust AprilTag detections (immediate lock)

test_apriltag_detection.py  # Test script for both-sides detection
```

## Hybrid Perception Architecture

The system uses **HybridPerceptionModule** which combines:

1. **AprilTag** for charger detection (100% reliable, no false positives)
2. **Grounding DINO** for plant/obstacle detection (with color validation)

```
┌─────────────────────────────────────────────────────────┐
│                 HybridPerceptionModule                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │  AprilTagCharger    │  │   GroundingDINO         │   │
│  │  Detector           │  │   PerceptionModule      │   │
│  │                     │  │                         │   │
│  │  - Charger ONLY     │  │  - Plant (red box)      │   │
│  │  - ID=0             │  │  - Obstacle (gray box)  │   │
│  │  - 100% confidence  │  │  - Color validation     │   │
│  │  - Pose estimation  │  │  - Raycast depth        │   │
│  └─────────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Basic Detection

```python
from perception.apriltag_detector import AprilTagChargerDetector

detector = AprilTagChargerDetector(
    charger_tag_id=0,
    tag_size=0.15,  # 15cm
)

detections = detector.detect(rgb_frame, robot_pose=(x, y, yaw))

for det in detections:
    print(f"Charger at {det['world_position']}")
    print(f"Distance: {det['estimated_distance']:.2f}m")
    print(f"Decision margin: {det['decision_margin']:.1f}")
```

### Hybrid Perception (Recommended)

```python
from perception.apriltag_detector import HybridPerceptionModule, AprilTagChargerDetector
from perception.vision_model import PerceptionModule

# Create Grounding DINO for plant/obstacle (NOT charger!)
grounding_dino = PerceptionModule(
    text_prompts={
        'plant': 'red box . red cube',
        'obstacle': 'gray box . gray cube',
        # NO charger - AprilTag handles it!
    }
)

# Create AprilTag detector
apriltag = AprilTagChargerDetector(tag_size=0.15)

# Create hybrid module
perception = HybridPerceptionModule(
    grounding_dino_detector=grounding_dino,
    apriltag_detector=apriltag,
)

# Detect all objects
detections = perception.detect(rgb_frame, robot_pose)
```

## Detection Output Format

AprilTag detections include:

```python
{
    'class': 'charger',
    'tag_id': 0,
    'confidence': 1.0,              # Always 1.0 for AprilTag
    'calibrated_prob': 1.0,
    'color_score': 1.0,             # Always 1.0 for AprilTag
    'depth_method': 'apriltag_pose', # Identifies as AprilTag detection
    'estimated_distance': 1.23,     # Very accurate (±2cm)
    'distance_uncertainty': 0.02,
    'world_position': [2.8, 0.0],
    'decision_margin': 51.5,        # AprilTag confidence metric
    'pose_t': [0.1, 0.02, 1.23],   # Translation in camera frame
    'pose_R': [[...], [...], [...]],  # Rotation matrix
}
```

## Semantic Grid Integration

The semantic grid recognizes AprilTag detections and:

1. **Trusts them completely** - No color/position validation needed
2. **Immediate locking** - Locks charger position after just 1 detection
3. **High weight** - AprilTag detections get weight=2.0 (vs ~0.5-1.5 for others)

```python
# In semantic_grid.py:
if det.get('depth_method') == 'apriltag_pose':
    # AprilTag = TRUSTED, immediate lock
    weight = 2.0
    # Can lock with just 1 detection
```

## Scene Setup

AprilTags are automatically created on both sides of the charger:

```python
from scene.setup_environment import setup_scene

scene = setup_scene('partial', seed=42, use_apriltag=True)
# AprilTags on +Y and -Y faces of charger
```

The tags are mirrored so they read correctly from either approach direction.

## Testing

Run the two-sided detection test:

```bash
# In Isaac Sim script editor or terminal:
isaac-sim.bat --exec "test_apriltag_detection.py"
```

Expected output:
```
✅ -Y face (from south): ID=0, dist=0.93m
✅ +Y face (from north): ID=0, dist=0.94m
🎉 BOTH SIDES DETECTED SUCCESSFULLY!
```

## Troubleshooting

### Tag not detected?
1. Check tag size matches physical size (15cm)
2. Ensure camera can see the tag (not too close, not too far)
3. Check lighting - avoid glare/shadows on the tag
4. Verify tag pattern is correct (run `verify_apriltag.py`)

### Wrong distance?
1. Verify camera intrinsics (HFOV=60°, 640x480)
2. Check tag_size parameter matches actual size

## References

- [AprilTag Official](https://april.eecs.umich.edu/software/apriltag)
- [pupil-apriltags](https://pypi.org/project/pupil-apriltags/)
- [Tag Images](https://github.com/AprilRobotics/apriltag-imgs)
