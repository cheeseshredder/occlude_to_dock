"""
Count re-scans from existing batch results.

Re-scan = information-gathering action (not committing to approach/dock)

Re-scan actions:
- EXPLORE, EXPLORE_TOWARD, EXPLORE_AROUND - exploring to find/verify target
- LOOK_AROUND - rotating to observe

Non-re-scan actions:
- APPROACH, APPROACH_AROUND - committing to target
- DOCK, DOCK_SUCCESS - docking
- INITIAL - starting state
- FORCED_BACKUP, BACKUP, STUCK_RECOVERY - recovery (not info-gathering)
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

# Actions that count as re-scans (information-gathering)
RESCAN_ACTIONS = {
    'EXPLORE',
    'EXPLORE_TOWARD', 
    'EXPLORE_AROUND',
    'LOOK_AROUND',
}

# Actions that are NOT re-scans
COMMIT_ACTIONS = {
    'APPROACH',
    'APPROACH_AROUND',
    'DOCK',
    'DOCK_SUCCESS',
    'INITIAL',
    'FORCED_BACKUP',
    'BACKUP',
    'STUCK_RECOVERY',
}


def extract_action_from_filename(filename: str) -> str:
    """Extract action name from belief map filename."""
    # Format: step_XX_ACTION.png or step_XX_ACTION_SUFFIX.png
    basename = os.path.basename(filename)
    # Remove extension
    name = os.path.splitext(basename)[0]
    # Remove step_XX_ prefix
    match = re.match(r'step_\d+_(.+)', name)
    if match:
        return match.group(1)
    return name


def count_rescans_in_trial(belief_maps: list) -> dict:
    """Count re-scans and other actions in a single trial."""
    actions = [extract_action_from_filename(f) for f in belief_maps]
    
    rescan_count = 0
    commit_count = 0
    other_count = 0
    action_counts = defaultdict(int)
    
    for action in actions:
        action_counts[action] += 1
        if action in RESCAN_ACTIONS:
            rescan_count += 1
        elif action in COMMIT_ACTIONS:
            commit_count += 1
        else:
            other_count += 1
            print(f"  Unknown action: {action}")
    
    return {
        'rescans': rescan_count,
        'commits': commit_count,
        'other': other_count,
        'action_counts': dict(action_counts),
        'actions_sequence': actions,
    }


def analyze_batch(batch_dir: str) -> dict:
    """Analyze a batch results directory."""
    results_file = os.path.join(batch_dir, 'results.json')
    summary_file = os.path.join(batch_dir, 'summary.json')
    
    if not os.path.exists(results_file):
        print(f"No results.json in {batch_dir}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    trial_rescans = []
    all_action_counts = defaultdict(int)
    successful_rescans = []
    failed_rescans = []
    
    for trial in results:
        belief_maps = trial.get('belief_maps', [])
        if not belief_maps:
            continue
        
        trial_stats = count_rescans_in_trial(belief_maps)
        trial_rescans.append(trial_stats['rescans'])
        
        # Track by success/failure
        if trial.get('success', False):
            successful_rescans.append(trial_stats['rescans'])
        else:
            failed_rescans.append(trial_stats['rescans'])
        
        # Aggregate action counts
        for action, count in trial_stats['action_counts'].items():
            all_action_counts[action] += count
    
    # Compute statistics
    rescans_array = np.array(trial_rescans)
    
    analysis = {
        'batch_dir': os.path.basename(batch_dir),
        'tier': summary.get('tier', 'unknown'),
        'total_trials': len(results),
        'success_rate': summary.get('success_rate', 0),
        'rescan_stats': {
            'mean': float(np.mean(rescans_array)) if len(rescans_array) > 0 else 0,
            'std': float(np.std(rescans_array)) if len(rescans_array) > 0 else 0,
            'median': float(np.median(rescans_array)) if len(rescans_array) > 0 else 0,
            'min': int(np.min(rescans_array)) if len(rescans_array) > 0 else 0,
            'max': int(np.max(rescans_array)) if len(rescans_array) > 0 else 0,
        },
        'successful_trials_rescans': {
            'mean': float(np.mean(successful_rescans)) if successful_rescans else 0,
            'median': float(np.median(successful_rescans)) if successful_rescans else 0,
        },
        'failed_trials_rescans': {
            'mean': float(np.mean(failed_rescans)) if failed_rescans else 0,
            'median': float(np.median(failed_rescans)) if failed_rescans else 0,
        },
        'action_totals': dict(all_action_counts),
        'per_trial_rescans': trial_rescans,
    }
    
    return analysis


def main():
    batch_base = r"C:\Users\abdul\Desktop\occlude_to_dock\batch_results"
    
    # Find latest heavy and partial batches
    batches = sorted(os.listdir(batch_base))
    
    # Get specific batches to analyze
    target_batches = [
        'batch_heavy_20251205_011302',   # 100 trials heavy
        'batch_partial_20251205_024212', # 25 trials partial
    ]
    
    print("=" * 70)
    print("RE-SCAN ANALYSIS FROM BATCH RESULTS")
    print("=" * 70)
    print()
    print("Re-scan actions (information-gathering):")
    print(f"  {RESCAN_ACTIONS}")
    print()
    print("Commit actions (approaching/docking):")
    print(f"  {COMMIT_ACTIONS}")
    print()
    
    for batch_name in target_batches:
        batch_path = os.path.join(batch_base, batch_name)
        if not os.path.exists(batch_path):
            print(f"Batch not found: {batch_name}")
            continue
        
        print("-" * 70)
        print(f"BATCH: {batch_name}")
        print("-" * 70)
        
        analysis = analyze_batch(batch_path)
        if analysis is None:
            continue
        
        print(f"Tier: {analysis['tier']}")
        print(f"Trials: {analysis['total_trials']}")
        print(f"Success Rate: {analysis['success_rate']*100:.1f}%")
        print()
        print("RE-SCAN STATISTICS:")
        stats = analysis['rescan_stats']
        print(f"  Mean:   {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"  Range:  [{stats['min']}, {stats['max']}]")
        print()
        print(f"  Successful trials: median={analysis['successful_trials_rescans']['median']:.1f}, mean={analysis['successful_trials_rescans']['mean']:.2f}")
        print(f"  Failed trials:     median={analysis['failed_trials_rescans']['median']:.1f}, mean={analysis['failed_trials_rescans']['mean']:.2f}")
        print()
        print("ACTION TOTALS:")
        for action, count in sorted(analysis['action_totals'].items(), key=lambda x: -x[1]):
            is_rescan = "📷" if action in RESCAN_ACTIONS else "  "
            print(f"  {is_rescan} {action}: {count}")
        print()
        
        # Check against target: median re-scans ≤ 2
        target_met = stats['median'] <= 2
        print(f"TARGET CHECK: Median re-scans ≤ 2")
        print(f"  Result: {stats['median']:.1f} {'✅ PASS' if target_met else '❌ FAIL'}")
        print()
        
        # Save analysis
        output_file = os.path.join(batch_path, 'rescan_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to: {output_file}")
        print()


if __name__ == '__main__':
    main()
