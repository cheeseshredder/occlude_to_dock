# Quick Start Guide - Occlude-to-Dock

## ✅ What's Been Created

The complete Occlude-to-Dock project has been set up with all core modules implemented:

### 📁 Project Structure

```
occlude_to_dock/
├── config/                      # ✅ Configuration files
│   ├── params.yaml             # ✅ All system parameters
│   ├── scene_partial.yaml      # ✅ Partial occlusion scene
│   └── calibration.yaml        # ✅ Calibration config (template)
│
├── perception/                  # ✅ Perception module
│   └── vision_model.py         # ✅ Grounding DINO + calibration
│
├── mapping/                     # ✅ Semantic mapping
│   └── semantic_grid.py        # ✅ Beta-Bernoulli + Dirichlet
│
├── planner/                     # ✅ Planning & control
│   ├── viewpoint_selector.py  # ✅ Information gain
│   ├── pomdp_policy.py        # ✅ QMDP planner
│   └── controller.py          # ✅ DWA + docking
│
├── evaluation/                  # ✅ Evaluation & logging
│   └── metrics_logger.py      # ✅ KPI tracking
│
├── calibration/                 # ✅ Calibration scripts
│   ├── collect_dataset.py     # ✅ Data collection
│   └── fit_temperature.py     # ✅ Temperature fitting
│
├── utils.py                    # ✅ Common utilities
├── main.py                     # ✅ Main orchestrator
├── test_system.py              # ✅ Test all modules
├── requirements.txt            # ✅ Dependencies
├── README.md                   # ✅ Full documentation
└── IMPLEMENTATION_PLAN.md      # ✅ Complete pseudo-code
```

## 🚀 Getting Started (10 minutes)

### Step 1: Verify Installation (2 min)

```bash
cd occlude_to_dock

# Test all modules
python test_system.py
```

**Expected output**: All 7 tests should pass ✓

### Step 2: Install Dependencies (2 min)

```bash
# Install Python packages
pip install -r requirements.txt

# Note: Isaac Sim 5.1 must be installed separately
```

### Step 3: Collect Calibration Data (2 min - MOCK MODE)

```bash
# This generates mock calibration data for testing
python calibration/collect_dataset.py --num_scenes 10 --samples_per_scene 20
```

### Step 4: Fit Temperature Calibration (1 min)

```bash
# Fit temperature parameter
python calibration/fit_temperature.py \
    --dataset calibration/calibration_dataset.pkl \
    --output config/calibration.yaml
```

### Step 5: Run Test Episode (3 min)

```bash
# Run single test episode with visualization
python main.py --mode test --seed 42 --visualize
```

## 📊 What Works Right Now (MOCK MODE)

The system currently runs in **MOCK MODE** which means:

✅ **Working**:
- All perception, mapping, planning modules function
- QMDP planner makes decisions based on belief
- Controllers compute velocities
- Full KPI logging and evaluation
- Belief heatmap snapshots

⚠️ **Mock Components**:
- Grounding DINO (falls back to mock detections)
- Isaac Sim integration (uses MockRobot class)
- Calibration data (randomly generated)

## 🔧 Next Steps for Full Implementation

### Immediate (Today):

1. **Isaac Sim Integration** (2-3 hours)
   - Create scene setup script in `scene/`
   - Connect to actual TurtleBot model
   - Implement camera interfaces
   - Add ground truth extraction

2. **Real Perception** (1 hour)
   - Verify Grounding DINO loads with CUDA
   - Test on Isaac Sim rendered frames
   - Collect real calibration dataset

3. **Testing** (2 hours)
   - Run end-to-end in Isaac Sim
   - Verify all modules integrate correctly
   - Debug any issues

### Tomorrow:

4. **Experiment Execution** (3-4 hours)
   - Run partial occlusion experiments (N=5 seeds)
   - Generate visualizations
   - Compute aggregate metrics

5. **Heavy Occlusion** (2 hours)
   - Create heavy occlusion scene config
   - Run experiments
   - Compare results

6. **Final Report** (2 hours)
   - Compile results
   - Generate plots and tables
   - Create demo video

## 🎯 Current System Capabilities

Even in mock mode, the system demonstrates:

### Belief-Aware Decision Making
```
Max belief: 0.65 → Decision: RESCAN (gather more info)
Max belief: 0.85 → Decision: APPROACH (high confidence)
```

### Information-Driven Exploration
- Generates viewpoint candidates around high-belief regions
- Ranks by expected information gain
- Moves to reduce uncertainty before committing

### Safe Navigation
- DWA computes collision-free trajectories
- Risk map inflates obstacles
- Path cost penalizes high-risk areas

### Complete Logging
- Episode data saved to JSON
- Belief snapshots as PNG images
- Aggregate metrics in CSV

## 📝 Configuration Tips

### Adjust Planning Thresholds

Edit `config/params.yaml`:

```yaml
planning:
  high_confidence_threshold: 0.8  # Lower to be more aggressive
  medium_confidence_threshold: 0.5
  max_rescans: 3                  # Increase for more caution
```

### Change Grid Resolution

```yaml
scene:
  grid_resolution: 0.05  # 5cm cells (current)
  # Try 0.10 for faster, or 0.025 for finer
```

### Tune Reward Weights

```yaml
planning:
  reward_dock: 100.0
  reward_step: -1.0
  lambda_risk: 0.5   # Increase to be more cautious
  lambda_ig: 2.0     # Increase to value information more
```

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Make sure you're in the project directory
cd occlude_to_dock

# Verify Python path
python -c "import sys; print(sys.path)"
```

### "Grounding DINO not loading"
- System will use mock detections automatically
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### "No module named 'transformers'"
```bash
pip install transformers torch torchvision
```

## 📖 Key Files to Review

1. **IMPLEMENTATION_PLAN.md** - Complete pseudo-code for all modules
2. **README.md** - Full project documentation
3. **config/params.yaml** - All tunable parameters
4. **main.py** - See how everything integrates

## 💡 Tips for Isaac Sim Integration

When you're ready to integrate with Isaac Sim:

1. Replace `MockRobot` class in `main.py` with actual Isaac Sim robot interface
2. Create `scene/setup_environment.py` following the pseudo-code in IMPLEMENTATION_PLAN.md
3. Use Isaac Sim's semantic segmentation for automatic ground truth
4. Connect camera annotators for RGB/depth/bounding boxes

Example structure:
```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
world = World()

# ... setup scene following scene_partial.yaml
```

## ✅ Success Criteria Checklist

When running experiments, verify:

- [ ] Success rate ≥ 85% (partial occlusion)
- [ ] Time to dock ≤ 60s (median)
- [ ] Rescans ≤ 2 (median)
- [ ] ECE ≤ 0.08 (after calibration)
- [ ] Zero collisions

## 🎉 You're Ready!

The entire pipeline is implemented and testable. You can:

1. ✅ Run the system end-to-end (mock mode)
2. ✅ Test all individual modules
3. ✅ Adjust parameters and see effects
4. ✅ Review complete pseudo-code
5. 🔧 Integrate with Isaac Sim when ready

Good luck with your project! 🚀
