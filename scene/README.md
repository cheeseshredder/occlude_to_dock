# Isaac Sim Scene Setup

This directory contains the complete Isaac Sim scene setup for the Occlude-to-Dock project.

## 📁 Files

```
scene/
├── setup_environment.py    ← Main scene setup (objects, lighting, etc.)
├── robot_interface.py      ← Robot control wrapper (differential drive)
└── README.md              ← This file
```

## 🚀 Quick Start

### Option 1: Run Test Script (Recommended)

```bash
# From project root
isaac-sim-python test_isaac_scene.py
```

This will:
- Create the complete scene
- Spawn all objects (charger, plant, distractors)
- Add robot with camera
- Test robot movement
- Verify camera and ground truth

### Option 2: Use in Python Script

```python
from scene.setup_environment import setup_scene
from scene.robot_interface import IsaacSimRobot

# Create scene
scene = setup_scene('partial', seed=42)

# Create robot interface
robot = IsaacSimRobot(
    scene['robot'],
    scene['world'],
    scene['camera']
)

# Use robot
pose = robot.get_pose()  # [x, y, theta]
robot.set_velocity(0.1, 0.0)  # Forward at 0.1 m/s

# Get sensor data
rgb, depth = robot.get_camera_data()

# Get ground truth
from scene.setup_environment import get_ground_truth
gt = get_ground_truth(scene['world'], scene['objects'], scene['camera'])
```

### Option 3: Isaac Sim Script Editor

Copy the contents of `test_isaac_scene.py` into Isaac Sim's Script Editor and run.

## 📋 Scene Components

### Objects (from config/scene_partial.yaml)

| Object | Position | Size/Radius | Color | Purpose |
|--------|----------|-------------|-------|---------|
| **Charger** | [2.8, 0.0, 0.5] | 0.3×0.2×0.4m | Red | Target for docking |
| **Plant** | [2.0, 0.0, 0.5] | r=0.2, h=1.0 | Green | Occluder (50% blocks charger) |
| **Distractor 1** | [2.0, 1.5, 0.5] | 0.35×0.25×0.45m | Gray | Confuser object |
| **Distractor 2** | [2.0, -1.5, 0.5] | 0.35×0.25×0.45m | Gray | Confuser object |
| **Robot** | [0.0, 0.0, 0.0] | TurtleBot | Blue | Mobile robot |

### Room
- **Size**: 6m × 6m
- **Walls**: 0.2m thick, 2.5m high
- **Floor**: Ground plane with collision

### Camera
- **Resolution**: 640×480
- **Position**: 20cm forward, 30cm up from robot base
- **Frequency**: 10 Hz
- **Sensors**: RGB, Depth, Semantic Segmentation

## 🎮 Robot Interface API

### IsaacSimRobot

```python
# Get current pose
pose = robot.get_pose()  # Returns [x, y, theta]

# Set velocity (differential drive)
robot.set_velocity(v_linear, v_angular)
# v_linear: m/s (positive = forward)
# v_angular: rad/s (positive = counter-clockwise)

# Stop
robot.stop()

# Reset to position
robot.reset(position=[0, 0, 0])

# Get camera data
rgb, depth = robot.get_camera_data()
# rgb: (480, 640, 3) uint8
# depth: (480, 640) float32
```

## 🔧 Configuration

Scene parameters are in `config/scene_partial.yaml`:

```yaml
room_size: [6.0, 6.0]

objects:
  charger:
    position: [2.8, 0.0, 0.5]
    size: [0.3, 0.2, 0.4]
    color: [1.0, 0.0, 0.0]
  # ... etc
```

To modify the scene, edit this file and re-run setup.

## 📊 Ground Truth Extraction

The scene provides automatic ground truth from semantic segmentation:

```python
from scene.setup_environment import get_ground_truth

gt = get_ground_truth(world, objects, camera)

# Returns:
{
    'charger_visible': bool,
    'charger_bbox': [x1, y1, x2, y2] or None,
    'charger_pose': [x, y, z],
    'plant_visible': bool,
    'plant_bbox': [x1, y1, x2, y2] or None,
    'plant_pose': [x, y, z],
    'distractors': [...]
}
```

This is used by Agent 2 for calibration dataset collection.

## 🔗 Integration with Main System

The main system (`main.py`) will use this scene instead of MockRobot:

```python
# OLD (mock mode):
# robot = MockRobot(config)

# NEW (Isaac Sim):
from scene.setup_environment import setup_scene
from scene.robot_interface import IsaacSimRobot

scene = setup_scene('partial', seed)
robot = IsaacSimRobot(scene['robot'], scene['world'], scene['camera'])
```

## ⚠️ Troubleshooting

### "Isaac Sim not available"
- **Solution**: Run script using Isaac Sim's Python: `isaac-sim-python script.py`
- **Alternative**: Copy-paste into Isaac Sim Script Editor

### "Could not load TurtleBot asset"
- **Solution**: Script will create simple robot base as fallback
- **Note**: This is OK for testing, may need proper TurtleBot USD later

### "Camera not configured"
- **Solution**: May need manual camera setup in Isaac Sim
- **Note**: Can still test with dummy camera data

### Objects falling through ground
- **Solution**: Already handled - all objects have `mass=0.0` (static)
- **Check**: Verify ground plane collision is enabled

### Robot not moving
- **Check**: Are wheel joint names correct?
- **Try**: Running with `headless=False` to see Isaac Sim window
- **Debug**: Add print statements in `set_velocity()`

## 📚 Related Files

- `config/scene_partial.yaml` - Scene configuration
- `config/params.yaml` - System parameters (camera, robot specs)
- `test_isaac_scene.py` - Complete test script
- `main.py` - Main system (will be integrated)

## 🔄 Next Steps for Integration

1. ✅ Scene setup complete (this directory)
2. ⏳ Wait for Agent 2 to complete calibration
3. ⏳ Integrate with main.py (replace MockRobot)
4. ⏳ Run full experiments

## 📞 Coordination with Agent 2

**What Agent 2 needs from this scene**:

1. `setup_scene(tier, seed)` - Creates scene
2. `get_camera_data(camera)` - Gets RGB/Depth
3. `get_ground_truth(world, objects, camera)` - Gets ground truth labels

**When ready**: Agent 2 will use these to collect calibration dataset.

## ✅ Status

- [x] Scene setup script created
- [x] Robot interface created
- [x] Test script created
- [x] Documentation complete
- [ ] Tested in actual Isaac Sim (requires Isaac Sim 5.1)
- [ ] Integrated with main system
- [ ] Used by Agent 2 for calibration

**Last updated**: 2024-12-02 (Agent 1 initial implementation)
