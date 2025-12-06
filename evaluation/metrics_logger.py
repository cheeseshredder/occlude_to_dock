"""
Evaluation Logger: Track and compute KPIs for each episode.
"""

import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime


class EvaluationLogger:
    """
    Track and compute KPIs for experiments.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize evaluation logger.
        
        Args:
            output_dir: Directory to save logs and metrics
        """
        self.output_dir = output_dir
        self.episode_data = []
        self.current_episode = None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'episodes'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'belief_snapshots'), exist_ok=True)
        
        print(f"EvaluationLogger initialized, saving to: {output_dir}")
    
    def start_episode(self, episode_id: int, seed: int, config: Dict):
        """Initialize logging for new episode."""
        self.current_episode = {
            'episode_id': episode_id,
            'seed': seed,
            'config': config,
            'start_time': time.time(),
            'detections': [],
            'actions': [],
            'belief_snapshots': [],
            'poses': [],
            'success': False,
            'time_to_dock': None,
            'num_rescans': 0,
            'risk_cost': 0.0,
            'path_length': 0.0,
            'final_pose': None
        }
        
        print(f"\n=== Episode {episode_id} Started (seed={seed}) ===")
    
    def log_detection(self, timestamp: float, detections: List[Dict]):
        """Log perception output."""
        if self.current_episode is None:
            return
        
        self.current_episode['detections'].append({
            'timestamp': timestamp,
            'detections': detections
        })
    
    def log_action(self, timestamp: float, action: Dict):
        """Log planner action."""
        if self.current_episode is None:
            return
        
        self.current_episode['actions'].append({
            'timestamp': timestamp,
            'action': action
        })
        
        if action['type'] == 'rescan':
            self.current_episode['num_rescans'] += 1
        
        print(f"  t={timestamp:.1f}s: Action={action['type']}")
    
    def log_pose(self, timestamp: float, pose: np.ndarray):
        """Log robot pose."""
        if self.current_episode is None:
            return
        
        self.current_episode['poses'].append({
            'timestamp': timestamp,
            'pose': pose.tolist()
        })
    
    def log_belief_snapshot(self, timestamp: float, belief_map: np.ndarray):
        """Save belief map snapshot as image."""
        if self.current_episode is None:
            return
        
        snapshot_dir = os.path.join(self.output_dir, 'belief_snapshots')
        snapshot_path = os.path.join(
            snapshot_dir,
            f"ep{self.current_episode['episode_id']}_t{timestamp:.2f}.png"
        )
        
        # Save as heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(belief_map, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Charger Belief')
        plt.title(f"Episode {self.current_episode['episode_id']}, t={timestamp:.2f}s")
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.savefig(snapshot_path, dpi=100)
        plt.close()
        
        self.current_episode['belief_snapshots'].append(snapshot_path)
    
    def update_path_metrics(self, prev_pose: np.ndarray, curr_pose: np.ndarray,
                           risk_map: np.ndarray, grid):
        """Update path length and risk cost."""
        if self.current_episode is None:
            return
        
        # Path length
        dist = np.sqrt(
            (curr_pose[0] - prev_pose[0])**2 +
            (curr_pose[1] - prev_pose[1])**2
        )
        self.current_episode['path_length'] += dist
        
        # Risk cost (integrate risk along path)
        i, j = grid.world_to_grid(curr_pose[:2])
        if grid.is_valid_cell(i, j):
            self.current_episode['risk_cost'] += risk_map[j, i] * dist
    
    def end_episode(self, success: bool, final_pose: np.ndarray):
        """Finalize episode logging."""
        if self.current_episode is None:
            return
        
        self.current_episode['success'] = success
        self.current_episode['end_time'] = time.time()
        self.current_episode['final_pose'] = final_pose.tolist()
        
        if success:
            self.current_episode['time_to_dock'] = (
                self.current_episode['end_time'] -
                self.current_episode['start_time']
            )
        
        # Save episode data to JSON
        episode_file = os.path.join(
            self.output_dir, 'episodes',
            f"episode_{self.current_episode['episode_id']}.json"
        )
        
        with open(episode_file, 'w') as f:
            json.dump(self.current_episode, f, indent=2)
        
        # Add to episode list
        self.episode_data.append(self.current_episode)
        
        # Print summary
        status = "SUCCESS" if success else "FAILED"
        print(f"\n=== Episode {self.current_episode['episode_id']} {status} ===")
        if success:
            print(f"  Time to dock: {self.current_episode['time_to_dock']:.2f}s")
        print(f"  Rescans: {self.current_episode['num_rescans']}")
        print(f"  Path length: {self.current_episode['path_length']:.2f}m")
        print(f"  Risk cost: {self.current_episode['risk_cost']:.3f}")
        
        self.current_episode = None
    
    def compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all episodes.
        
        Returns:
            metrics: Dict with mean±std for all KPIs
        """
        if len(self.episode_data) == 0:
            return {}
        
        # Success rate
        successes = [ep['success'] for ep in self.episode_data]
        success_rate = np.mean(successes)
        
        # Time to dock (for successful episodes only)
        times = [ep['time_to_dock'] for ep in self.episode_data if ep['success']]
        if len(times) > 0:
            mean_time = np.mean(times)
            std_time = np.std(times)
        else:
            mean_time = None
            std_time = None
        
        # Re-scans
        rescans = [ep['num_rescans'] for ep in self.episode_data]
        median_rescans = np.median(rescans)
        mean_rescans = np.mean(rescans)
        
        # Risk cost
        risk_costs = [ep['risk_cost'] for ep in self.episode_data]
        mean_risk = np.mean(risk_costs)
        std_risk = np.std(risk_costs)
        
        # Path length
        path_lengths = [ep['path_length'] for ep in self.episode_data]
        mean_path = np.mean(path_lengths)
        std_path = np.std(path_lengths)
        
        metrics = {
            'num_episodes': len(self.episode_data),
            'success_rate': float(success_rate),
            'mean_time_to_dock': float(mean_time) if mean_time is not None else None,
            'std_time_to_dock': float(std_time) if std_time is not None else None,
            'median_rescans': float(median_rescans),
            'mean_rescans': float(mean_rescans),
            'mean_risk_cost': float(mean_risk),
            'std_risk_cost': float(std_risk),
            'mean_path_length': float(mean_path),
            'std_path_length': float(std_path),
        }
        
        return metrics
    
    def save_summary(self):
        """Save aggregate metrics to CSV."""
        metrics = self.compute_aggregate_metrics()
        
        if not metrics:
            print("No episodes logged yet")
            return
        
        summary_file = os.path.join(self.output_dir, "summary.csv")
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        
        print(f"\n=== Aggregate Metrics ===")
        print(f"Episodes: {metrics['num_episodes']}")
        print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
        if metrics['mean_time_to_dock'] is not None:
            print(f"Time to Dock: {metrics['mean_time_to_dock']:.2f}±{metrics['std_time_to_dock']:.2f}s")
        print(f"Rescans: {metrics['median_rescans']:.0f} (median), {metrics['mean_rescans']:.1f} (mean)")
        print(f"Risk Cost: {metrics['mean_risk_cost']:.3f}±{metrics['std_risk_cost']:.3f}")
        print(f"Path Length: {metrics['mean_path_length']:.2f}±{metrics['std_path_length']:.2f}m")
        print(f"\nSummary saved to: {summary_file}")


def compute_ECE_from_episodes(episode_data: List[Dict], num_bins: int = 15) -> Dict:
    """
    Compute Expected Calibration Error from logged detections.
    
    Args:
        episode_data: List of episode dicts
        num_bins: Number of bins for calibration curve
        
    Returns:
        Dict with ECE and reliability diagram data
    """
    all_probs = []
    all_labels = []
    
    for episode in episode_data:
        for det_log in episode['detections']:
            for det in det_log['detections']:
                if det['class'] == 'charger':
                    all_probs.append(det['calibrated_prob'])
                    
                    # Check if correct (would need ground truth)
                    is_correct = det.get('is_correct', None)
                    if is_correct is not None:
                        all_labels.append(1 if is_correct else 0)
    
    if len(all_probs) == 0 or len(all_labels) == 0:
        return {'ECE': None, 'reliability_diagram': None}
    
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    
    # Compute ECE
    from perception.vision_model import compute_ECE
    ECE = compute_ECE(probs, labels, num_bins)
    
    # Compute reliability diagram data
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for m in range(num_bins):
        in_bin = (probs >= bin_boundaries[m]) & (probs < bin_boundaries[m+1])
        
        if np.sum(in_bin) > 0:
            bin_confidences.append(float(np.mean(probs[in_bin])))
            bin_accuracies.append(float(np.mean(labels[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_confidences.append(None)
            bin_accuracies.append(None)
            bin_counts.append(0)
    
    return {
        'ECE': float(ECE),
        'reliability_diagram': {
            'confidences': bin_confidences,
            'accuracies': bin_accuracies,
            'counts': bin_counts
        }
    }


def plot_reliability_diagram(reliability_data: Dict, output_path: str):
    """Plot and save reliability diagram."""
    if reliability_data['reliability_diagram'] is None:
        print("No reliability data available")
        return
    
    rd = reliability_data['reliability_diagram']
    confidences = [c for c in rd['confidences'] if c is not None]
    accuracies = [a for a in rd['accuracies'] if a is not None]
    
    plt.figure(figsize=(8, 8))
    
    # Plot bars
    x_pos = np.arange(len(confidences))
    plt.bar(x_pos, accuracies, width=0.8, alpha=0.7, label='Accuracy in bin')
    
    # Plot confidence line
    plt.plot(x_pos, confidences, 'ro-', label='Average confidence', linewidth=2)
    
    # Plot perfect calibration line
    plt.plot(x_pos, x_pos / len(x_pos), 'k--', label='Perfect calibration', linewidth=2)
    
    plt.xlabel('Confidence Bin', fontsize=12)
    plt.ylabel('Accuracy / Confidence', fontsize=12)
    plt.title(f'Reliability Diagram (ECE={reliability_data["ECE"]:.4f})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Reliability diagram saved to: {output_path}")


if __name__ == "__main__":
    print("Testing EvaluationLogger...")
    
    # Create logger
    logger = EvaluationLogger('/tmp/test_occlude_to_dock')
    
    # Simulate episode
    logger.start_episode(0, 42, {'occlusion': 'partial'})
    
    # Log some events
    logger.log_action(0.5, {'type': 'move', 'target': (1, 0, 0)})
    logger.log_action(1.0, {'type': 'rescan'})
    logger.log_action(2.0, {'type': 'approach', 'target': (2, 0, 0)})
    
    # End episode
    logger.end_episode(True, np.array([2.0, 0.0, 0.0]))
    
    # Compute metrics
    metrics = logger.compute_aggregate_metrics()
    print(f"\nMetrics: {metrics}")
    
    # Save summary
    logger.save_summary()
    
    print("\n✓ EvaluationLogger tests passed!")
