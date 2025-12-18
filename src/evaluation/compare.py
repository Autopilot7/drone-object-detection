"""
Comprehensive comparison of different approaches
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

from .st_iou import evaluate_dataset
from .visualize import plot_comparison, plot_st_iou_results


class ResultsComparator:
    """
    Compare results from multiple approaches
    """
    
    def __init__(self):
        """Initialize comparator"""
        self.results = {}
        self.ground_truth = None
    
    def add_results(
        self,
        approach_name: str,
        predictions: Dict[str, List],
        ground_truth: Dict[str, List] = None
    ):
        """
        Add results for an approach
        
        Args:
            approach_name: Name of the approach
            predictions: Predictions dictionary
            ground_truth: Ground truth dictionary (if not already set)
        """
        if ground_truth is not None and self.ground_truth is None:
            self.ground_truth = ground_truth
        
        # Evaluate
        mean_st_iou, per_video_scores = evaluate_dataset(
            self.ground_truth,
            predictions
        )
        
        self.results[approach_name] = {
            'mean_st_iou': mean_st_iou,
            'per_video_scores': per_video_scores,
            'predictions': predictions
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Get summary table of all approaches
        
        Returns:
            Pandas DataFrame with summary statistics
        """
        data = []
        
        for approach_name, result in self.results.items():
            per_video = result['per_video_scores']
            scores = list(per_video.values())
            
            data.append({
                'Approach': approach_name,
                'Mean ST-IoU': result['mean_st_iou'],
                'Std ST-IoU': np.std(scores),
                'Min ST-IoU': np.min(scores),
                'Max ST-IoU': np.max(scores),
                'Median ST-IoU': np.median(scores)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Mean ST-IoU', ascending=False)
    
    def plot_comparison(self):
        """Plot comparison of all approaches"""
        per_video_dict = {
            name: result['per_video_scores']
            for name, result in self.results.items()
        }
        
        plot_comparison(per_video_dict, title="Approach Comparison")
    
    def plot_per_category(self):
        """Plot performance by object category"""
        # Collect data by category
        category_data = {}
        
        for approach_name, result in self.results.items():
            per_video = result['per_video_scores']
            
            for video_id, score in per_video.items():
                category = '_'.join(video_id.split('_')[:-1])
                
                if category not in category_data:
                    category_data[category] = {}
                
                if approach_name not in category_data[category]:
                    category_data[category][approach_name] = []
                
                category_data[category][approach_name].append(score)
        
        # Compute mean per category
        categories = sorted(category_data.keys())
        approaches = list(self.results.keys())
        
        data = []
        for category in categories:
            for approach in approaches:
                scores = category_data[category].get(approach, [0])
                mean_score = np.mean(scores)
                data.append({
                    'Category': category,
                    'Approach': approach,
                    'Mean ST-IoU': mean_score
                })
        
        df = pd.DataFrame(data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_pivot = df.pivot(index='Category', columns='Approach', values='Mean ST-IoU')
        df_pivot.plot(kind='bar', ax=ax)
        
        ax.set_title('Performance by Object Category')
        ax.set_ylabel('Mean ST-IoU')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Approach')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def save_report(self, output_path: str):
        """
        Save comprehensive report
        
        Args:
            output_path: Output directory path
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary table
        summary = self.get_summary_table()
        summary.to_csv(output_dir / "summary.csv", index=False)
        
        # Save detailed results
        detailed_results = {}
        for approach_name, result in self.results.items():
            detailed_results[approach_name] = {
                'mean_st_iou': float(result['mean_st_iou']),
                'per_video_scores': {
                    k: float(v) for k, v in result['per_video_scores'].items()
                }
            }
        
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Report saved to {output_dir}")
    
    def print_summary(self):
        """Print summary to console"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        summary = self.get_summary_table()
        print(summary.to_string(index=False))
        
        print("\n" + "="*60)
        print("BEST APPROACH: {}".format(summary.iloc[0]['Approach']))
        print("Mean ST-IoU: {:.4f}".format(summary.iloc[0]['Mean ST-IoU']))
        print("="*60 + "\n")

