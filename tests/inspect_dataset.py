import pickle
import numpy as np
import sys

def inspect_dataset(filepath):
    print(f"Inspecting {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total samples: {len(data)}")
    
    charger_visible_count = 0
    valid_bbox_count = 0
    
    for i, sample in enumerate(data[:10]): # Check first 10
        gt = sample['ground_truth']
        print(f"\nSample {i}:")
        print(f"  Charger Visible: {gt['charger_visible']}")
        print(f"  Charger BBox: {gt['charger_bbox']}")
        print(f"  Charger Pose: {gt['charger_pose']}")
        
        if gt['charger_visible']:
            charger_visible_count += 1
            if gt['charger_bbox'] is not None:
                valid_bbox_count += 1
                # Check bounds
                bbox = gt['charger_bbox']
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > 640 or bbox[3] > 480:
                    print("  ⚠ WARNING: BBox out of bounds!")
    
    # Count totals
    total_visible = sum(1 for s in data if s['ground_truth']['charger_visible'])
    total_bbox = sum(1 for s in data if s['ground_truth']['charger_bbox'] is not None)
    
    print(f"\nSummary:")
    print(f"  Total Visible: {total_visible}/{len(data)}")
    print(f"  Total Valid BBoxes: {total_bbox}/{len(data)}")

if __name__ == "__main__":
    inspect_dataset('calibration/calibration_dataset.pkl')
