# Occlude-to-Dock: Uncertainty-Aware Language-Guided Charging-Dock Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-2023.1.1+-green.svg)](https://developer.nvidia.com/isaac-sim)

An uncertainty-aware vision-language-action pipeline for autonomous charging-dock discovery under occlusion. The system parses natural language instructions, uses calibrated open-vocabulary detection, maintains probabilistic semantic maps, and employs QMDP planning to navigate and dock even when the charger is hidden behind obstacles.

## Overview

Mobile robots often fail to locate charging docks when targets are occluded by furniture or clutter. Traditional approaches commit to approach actions based on raw detection scores, leading to failures when confidence is miscalibrated.

**Occlude-to-Dock** addresses this by:
- Parsing language instructions into structured goals with constraint propagation
- Calibrating detection probabilities via Platt scaling (ECE: 0.336 → 0.070)
- Maintaining belief over charger locations using Bayesian semantic mapping
- Selecting actions via QMDP planning that balances exploration and exploitation

## Results

Evaluated on 100 trials per occlusion tier with 0.40m docking threshold:

| Scenario | Success Rate | Target | Time-to-Dock | Final Distance |
|----------|-------------|--------|--------------|----------------|
| **Partial Occlusion** | **92%** | ≥85% ✓ | 33.1 ± 5.2s | 0.332m |
| **Heavy Occlusion** | **73%** | ≥65% ✓ | 45.1 ± 7.3s | 0.339m |

**Calibration:** Expected Calibration Error reduced from 0.336 to **0.070** (79% improvement)

## System Architecture

```
Language Instruction
        ↓
┌─────────────────┐
│ Goal Interpreter│ → Extracts target, relation, constraints
└────────┬────────┘
         ↓
┌─────────────────┐
│   Perception    │ → Grounding DINO + Platt scaling calibration
└────────┬────────┘
         ↓
┌─────────────────┐
│ Semantic Fusion │ → Beta-Bernoulli occupancy + Dirichlet semantics
└────────┬────────┘
         ↓
┌─────────────────┐
│  QMDP Planner   │ → Belief-based Q-value action selection
└────────┬────────┘
         ↓
┌─────────────────┐
│   Controller    │ → Motion execution + AprilTag docking
└─────────────────┘
```

## Key Features

- **Natural Language Parsing**: Extracts target, spatial relations, and constraints from instructions like "dock at the charger behind the plant"
- **Open-Vocabulary Detection**: Grounding DINO for text-conditioned object detection
- **Calibrated Perception**: Platt scaling ensures detection probabilities match empirical accuracy
- **Probabilistic Semantic Mapping**: Per-cell belief maintenance with Beta-Bernoulli occupancy and Dirichlet-Multinomial semantics
- **QMDP Planning**: True belief-based action selection over {DOCK, APPROACH, EXPLORE, RESCAN, ROTATE}
- **AprilTag Docking**: Precise terminal alignment using fiducial markers

## Technical Details

### Semantic Mapping

Each grid cell maintains:
- **Occupancy posterior**: Beta distribution $(\alpha_c, \beta_c)$
- **Semantic posterior**: Dirichlet distribution $\eta_c \in \mathbb{R}^K_{>0}$

Bayesian updates for detection with weight $w_c$:

$$\alpha_c \leftarrow \alpha_c + w_c \cdot o_c$$

$$\beta_c \leftarrow \beta_c + w_c \cdot (1 - o_c)$$

$$\eta_{c,k} \leftarrow \eta_{c,k} + w_c \cdot p_{i,k}$$

Charger belief map:

$$b(c) = \frac{\pi_{c,\text{charger}}}{\sum_{c'} \pi_{c',\text{charger}}}$$

### QMDP Planning

QMDP approximates POMDP by assuming uncertainty resolves after one action:

$$a^* = \arg\max_a \sum_s b(s) \cdot Q^*(s, a)$$

**Action Space:**
- `DOCK`: Final docking (high reward if at charger)
- `APPROACH`: Move toward highest-belief cell
- `EXPLORE`: Maximize expected information gain
- `RESCAN`: Re-observe to update belief
- `ROTATE`: Face highest-belief direction

**Reward Function:**

$$R(s, b, a) = R_{\text{dock}} \cdot \mathbb{I}_{\{a = \text{dock}\}} - \lambda_{\text{risk}} \cdot \text{Risk}(s, a) - \lambda_{\text{step}} + \lambda_{\text{ig}} \cdot \text{IG}(a)$$

### Perception Calibration

Raw Grounding DINO confidences are calibrated via Platt scaling:

$$p_i = \sigma(a \cdot z_i + b)$$

Where $z_i = \log\frac{c_i}{1 - c_i}$ and $(a, b) = (1.51, 1.60)$ are learned parameters.

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{m=1}^M \frac{n_m}{\sum_j n_j} \left| \hat{a}_m - \hat{c}_m \right|$$

## Installation

### Prerequisites

- NVIDIA Isaac Sim 2023.1.1+
- Python 3.10+
- CUDA 11.x+
- RTX GPU (tested on RTX 4070 Ti SUPER)

### Setup

```bash
git clone https://github.com/yourusername/occlude_to_dock.git
cd occlude_to_dock
pip install -r requirements.txt
```

### Isaac Sim Python Environment

Use the Isaac Sim Python environment for running simulations:

```bash
# Windows
.\isaac-python.bat main.py

# Or set up Isaac Sim Python path
```

## Usage

### Quick Start

```bash
# Run single episode with partial occlusion
python main.py --config config/scene_partial.yaml

# Run with heavy occlusion
python main.py --config config/scene_heavy.yaml
```

### Batch Testing

```bash
# Run 100 trials with QMDP planner
python batch_test_qmdp_true.py --scenario partial --trials 100 --seed 42

python batch_test_qmdp_true.py --scenario heavy --trials 100 --seed 42
```

### Language Instructions

The system responds to natural language commands:

| Instruction | Parsed Config |
|-------------|---------------|
| "dock at the charger behind the plant" | conf=0.6, expect_occlusion=True |
| "go to the charging station near the obstacle" | conf=0.4, expect_occlusion=False |
| "find the dock occluded by the plant" | conf=0.6, expect_occlusion=True |

### Demo Recording

```bash
python run_demo.py --scenario partial --save-frames
python visualization/create_video.py --input demo_frames/ --output demo.mp4
```

## Project Structure

```
occlude_to_dock/
├── config/                 # YAML configuration files
│   ├── params.yaml         # Global parameters
│   ├── calibration.yaml    # Calibration settings
│   ├── scene_partial.yaml  # Partial occlusion scene
│   └── scene_heavy.yaml    # Heavy occlusion scene
├── perception/             # Vision and detection
│   ├── vision_model.py     # Grounding DINO wrapper
│   ├── apriltag_detector.py# AprilTag detection
│   ├── depth_sensor.py     # Depth processing
│   └── synthetic_depth.py  # Simulated depth
├── mapping/                # Semantic grid
│   └── semantic_grid.py    # Bayesian belief mapping
├── planner/                # Planning and control
│   ├── qmdp_planner_real.py# TRUE QMDP implementation
│   ├── pomdp_planner.py    # POMDP interface
│   ├── controller.py       # Motion controller
│   └── viewpoint_selector.py
├── language/               # NLP parsing
│   └── parser.py           # Instruction parser
├── scene/                  # Isaac Sim setup
│   ├── setup_environment.py
│   ├── robot_interface.py
│   └── apriltag_setup.py
├── calibration/            # Perception calibration
│   ├── run_calibration.py
│   ├── collect_dataset.py
│   └── fit_temperature.py
├── evaluation/             # Metrics and logging
│   ├── metrics_logger.py
│   └── calibration_metrics.py
├── visualization/          # Visualization tools
│   ├── create_video.py
│   └── realtime_heatmap.py
├── docs/                   # Documentation
│   └── APRILTAG_INTEGRATION.md
├── main.py                 # Main entry point
├── batch_test_qmdp_true.py # Batch evaluation
└── requirements.txt
```

## Configuration

### Key Parameters (config/params.yaml)

```yaml
# QMDP Planner
qmdp:
  reward_dock: 100.0
  reward_step: -1.0
  lambda_risk: 0.5
  lambda_ig: 2.0
  confidence_high: 0.7
  confidence_med: 0.4
  confidence_low: 0.15

# Perception
perception:
  detection_threshold: 0.2
  calibration_a: 1.51
  calibration_b: 1.60

# Docking
docking:
  success_distance: 0.40  # meters
  apriltag_size: 0.06     # 6cm tags
```

### Scene Configuration

Partial occlusion (`config/scene_partial.yaml`):
- Robot start: (0.0, 0.7)
- Charger: (2.8, 0.0)
- Plant partially blocks view

Heavy occlusion (`config/scene_heavy.yaml`):
- Robot start: (0.0, 2.0)  
- Charger: (2.8, 0.0)
- Plant fully blocks initial view

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Success Rate | Episodes reaching ≤0.40m of charger | ≥85% (partial), ≥65% (heavy) |
| Time-to-Dock | Seconds from start to success | ≤60s (partial), ≤90s (heavy) |
| Final Distance | Euclidean distance at termination | <0.40m for success |
| ECE | Expected Calibration Error | ≤0.08 |

## QMDP Action Distribution

From 100 trials per scenario:

| Action | Partial | Heavy |
|--------|---------|-------|
| APPROACH | 761 | 775 |
| DOCK | 92 | 73 |
| EXPLORE | 19 | 19 |
| BACKUP | 5 | 5 |

## Troubleshooting

### CUDA/GPU Issues
- Ensure CUDA 11.x is installed and compatible with your GPU
- Isaac Sim requires RTX GPU with recent drivers

### Grounding DINO
- Model weights download automatically on first run
- Requires ~2GB VRAM for inference

### AprilTag Detection
- Ensure proper camera calibration (47° HFOV)
- Tags must be well-lit and within 0.2-2.0m range

## References

1. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection," ECCV 2024
2. Gan et al., "Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping," IEEE RA-L 2020
3. Littman et al., "Learning policies for partially observable environments: Scaling up," ICML 1995
4. Olson, "AprilTag: A Robust and Flexible Visual Fiducial System," ICRA 2011

## Acknowledgments

- Course: Cognitive Robotics, Khalifa University
- Simulation: NVIDIA Isaac Lab
- Detection: Grounding DINO (IDEA Research)

## Author

**Abdelrahman Awwad**  
Khalifa University  
100067728@ku.ac.ae
