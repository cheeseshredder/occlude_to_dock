# Examples and Results

This folder contains sample outputs and evaluation results from the Occlude-to-Dock system.

## Contents

### `results_summary.json`
Complete evaluation metrics from 100 trials per occlusion scenario:
- **Partial Occlusion**: 92% success rate (target: ≥85%)
- **Heavy Occlusion**: 73% success rate (target: ≥65%)
- **Calibration**: ECE reduced from 0.336 → 0.070 (79% improvement)

### `demo_partial/`
Sample frames from a successful partial occlusion trial showing the QMDP planner in action:
- `step_00_INITIAL.png` - Starting position, charger partially visible
- `step_04_APPROACH.png` - Mid-navigation with belief heatmap
- `step_08_DOCK_COMPLETE.png` - Successful docking

### `demo_heavy/`
Sample frames from a successful heavy occlusion trial:
- `step_00_INITIAL.png` - Starting position, charger fully hidden
- `step_01_EXPLORE.png` - Exploration action to gain information
- `step_06_APPROACH.png` - Approaching after localizing charger
- `step_11_DOCK_COMPLETE.png` - Successful docking

### `calibration_results.yaml`
Platt scaling calibration parameters and metrics.

## Key Metrics

| Scenario | Success | Time-to-Dock | Final Distance |
|----------|---------|--------------|----------------|
| Partial  | 92%     | 33.1 ± 5.1s  | 0.332m        |
| Heavy    | 73%     | 45.1 ± 7.3s  | 0.339m        |

## QMDP Action Distribution

| Action   | Partial | Heavy |
|----------|---------|-------|
| APPROACH | 761     | 775   |
| DOCK     | 92      | 73    |
| EXPLORE  | 19      | 19    |
| BACKUP   | 5       | 5     |
