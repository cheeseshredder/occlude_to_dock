"""
Verify calibration dataset contains real data (not mock).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
dataset_path = 'calibration/calibration_dataset.pkl'

print("Loading dataset...")
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

print(f"✓ Loaded {len(data)} samples\n")

# Check first sample
sample = data[0]

print("Sample 0 structure:")
for key in sample.keys():
    value = sample[key]
    if isinstance(value, np.ndarray):
        print(f"  {key}: {value.shape} {value.dtype}")
    else:
        print(f"  {key}: {type(value)}")

print("\nChecking if data is REAL or MOCK...")

# Check RGB image
rgb = sample['rgb']
unique_values = len(np.unique(rgb))
print(f"  RGB unique values: {unique_values}")

if unique_values > 100000:
    print("  ✓ REAL data (high diversity)")
    is_real = True
else:
    print("  ⚠ Might be MOCK data (too uniform)")
    is_real = False

# Check ground truth
gt = sample['ground_truth']
print(f"\n  Ground truth keys: {gt.keys()}")
print(f"  Charger visible: {gt['charger_visible']}")
print(f"  Charger bbox: {gt['charger_bbox']}")
print(f"  Charger pose: {gt['charger_pose']}")

# Visualize first 3 samples
print("\nVisualizing samples...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(min(3, len(data))):
    sample = data[i]
    rgb = sample['rgb']
    gt = sample['ground_truth']
    
    axes[i].imshow(rgb)
    axes[i].set_title(f"Sample {i}\nCharger visible: {gt['charger_visible']}")
    axes[i].axis('off')
    
    # Draw bbox if available
    if gt['charger_bbox'] is not None:
        bbox = gt['charger_bbox']
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                        linewidth=2, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)

plt.tight_layout()
plt.savefig('calibration_data_preview.png', dpi=150)
print("✓ Saved visualization to: calibration_data_preview.png")

# Summary
print("\n" + "="*60)
if is_real:
    print("✅ Dataset contains REAL data from Isaac Sim!")
    print("   Ready for temperature calibration.")
else:
    print("⚠️  Dataset might be MOCK data")
    print("   Check Isaac Sim was running during collection.")
print("="*60)
