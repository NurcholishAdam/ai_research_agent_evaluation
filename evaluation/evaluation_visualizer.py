#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Results Visualizer
Creates comprehensive visualizations for RAG vs Graph-Aware Retrieval comparison
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EvaluationVisualizer:
    """Creates visualizations for evaluation results"""
    
    def __init__(self, results_path: str):
        """Initialize with evaluation results"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.pure_rag_metrics = self.results["pure_rag"]["metrics"]
        self.graph_rag_metrics = self.results["graph_aware_rag"]["metrics"]
        self.improvements = self.results["improvements"]
        
        # Create output directory
        self.output_dir = os.path.dirname(results_path)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def create_recall_comparison(self):
        """Create recall@k comparison chart"""
        
        k_values = [1, 3, 5, 10]
        pure_rag_recalls = [self.pure_rag_metrics["recall_at_k"][str(k)] for k in k_values]
        graph_rag_recalls = [self.graph_rag_metrics["recall_at_k"][str(k)] for k in k_values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pure_rag_recalls, width, label='Pure RAG', alpha=0.8)
        bars2 = ax.bar(x + width/2, graph_rag_recalls, width, label='Graph-Aware RAG', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('k (Top-k Results)')
        ax.set_ylabel('Recall@k')
        ax.set_title('Recall@k Comparison: Pure RAG vs Graph-Aware RAG')
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'recall_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_radar_chart(self):
        """Create radar chart comparing all metrics"""
        
        # Metrics to include (normalized to 0-1 scale)
        metrics = ['Recall@5', 'MRR', 'F1 Score', 'Exact Match', 'Low Hallucination']
        
        pure_values = [
            self.pure_rag_metrics["recall_at_k"]["5"],
            self.pure_rag_metrics["mrr"],
            self.pure_rag_metrics["f1_score"],
            self.pure_rag_metrics["exact_match"],
            1.0 - self.pure_rag_metrics["hallucination_rate"]  # Invert for "low hallucination"
        ]
        
        graph_values = [
            self.graph_rag_metrics["recall_at_k"]["5"],
            self.graph_rag_metrics["mrr"],
            self.graph_rag_metrics["f1_score"],
            self.graph_rag_metrics["exact_match"],
            1.0 - self.graph_rag_metrics["hallucination_rate"]
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        pure_values += pure_values[:1]
        graph_values += graph_values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, pure_values, 'o-', linewidth=2, label='Pure RAG', alpha=0.8)
        ax.fill(angles, pure_values, alpha=0.25)
        
        ax.plot(angles, graph_values, 'o-', linewidth=2, label='Graph-Aware RAG', alpha=0.8)
        ax.fill(angles, graph_values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Comparison: All Metrics\n(Higher is Better)', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'metrics_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_improvement_chart(self):
        """Create improvement percentage chart"""
        
        # Filter out improvements with reasonable values
        filtered_improvements = {}
        for metric, improvement in self.improvements.items():
            if abs(improvement) < 1000:  # Filter out extreme values
                filtered_improvements[metric] = improvement
        
        if not filtered_improvements:
            print("No reasonable improvement values to plot")
            return
        
        metrics = list(filtered_improvements.keys())
        improvements = list(filtered_improvements.values())
        
        # Create color map (green for positive, red for negative)
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(metrics, improvements, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            width = bar.get_width()
            ax.annotate(f'{improvement:.1f}%',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5 if width >= 0 else -5, 0),
                       textcoords="offset points",
                       ha='left' if width >= 0 else 'right',
                       va='center', fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Improvement (%)')
        ax.set_title('Graph-Aware RAG Improvements over Pure RAG\n(Positive = Better)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Format metric names
        formatted_metrics = [metric.replace('_', ' ').title() for metric in metrics]
        ax.set_yticklabels(formatted_metrics)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'improvements.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_comparison(self):
        """Create performance (timing) comparison"""
        
        categories = ['Retrieval Time', 'Generation Time']
        pure_times = [
            self.pure_rag_metrics["avg_retrieval_time"],
            self.pure_rag_metrics["avg_generation_time"]
        ]
        graph_times = [
            self.graph_rag_metrics["avg_retrieval_time"],
            self.graph_rag_metrics["avg_generation_time"]
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pure_times, width, label='Pure RAG', alpha=0.8)
        bars2 = ax.bar(x + width/2, graph_times, width, label='Graph-Aware RAG', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}s',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Operation Type')
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Performance Comparison: Processing Times')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_hyperparameter_tuning_heatmap(self):
        """Create heatmap for hyperparameter tuning results"""
        
        if not self.results.get("hyperparameter_tuning"):
            print("No hyperparameter tuning results to visualize")
            return
        
        tuning_results = self.results["hyperparameter_tuning"]["all_results"]
        
        # Create pivot table for heatmap
        df = pd.DataFrame(tuning_results)
        pivot_table = df.pivot(index='depth', columns='alpha', values='combined_score')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Combined Score'})
        
        ax.set_title('Hyperparameter Tuning Results\n(Graph-Vector Weight α vs Expansion Depth)')
        ax.set_xlabel('Graph-Vector Weight (α)')
        ax.set_ylabel('Expansion Depth')
        
        # Mark best configuration
        best_config = self.results["hyperparameter_tuning"]["best_config"]
        best_alpha_idx = list(pivot_table.columns).index(best_config["alpha"])
        best_depth_idx = list(pivot_table.index).index(best_config["depth"])
        
        ax.add_patch(plt.Rectangle((best_alpha_idx, best_depth_idx), 1, 1, 
                                  fill=False, edgecolor='blue', lw=3))
        ax.text(best_alpha_idx + 0.5, best_depth_idx + 0.5, '★', 
               ha='center', va='center', fontsize=20, color='blue')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'hyperparameter_tuning.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_metrics_breakdown(self):
        """Create detailed breakdown of all metrics"""
        
        # Prepare data
        metrics_data = {
            'Metric': [],
            'Pure RAG': [],
            'Graph-Aware RAG': [],
            'Improvement (%)': []
        }
        
        # Recall metrics
        for k in [1, 3, 5, 10]:
            metrics_data['Metric'].append(f'Recall@{k}')
            metrics_data['Pure RAG'].append(self.pure_rag_metrics["recall_at_k"][str(k)])
            metrics_data['Graph-Aware RAG'].append(self.graph_rag_metrics["recall_at_k"][str(k)])
            metrics_data['Improvement (%)'].append(self.improvements.get(f'recall_at_{k}', 0))
        
        # Other metrics
        other_metrics = [
            ('MRR', 'mrr'),
            ('F1 Score', 'f1_score'),
            ('Exact Match', 'exact_match'),
            ('Hallucination Rate', 'hallucination_rate')
        ]
        
        for display_name, key in other_metrics:
            metrics_data['Metric'].append(display_name)
            metrics_data['Pure RAG'].append(self.pure_rag_metrics[key])
            metrics_data['Graph-Aware RAG'].append(self.graph_rag_metrics[key])
            metrics_data['Improvement (%)'].append(self.improvements.get(key, 0))
        
        df = pd.DataFrame(metrics_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.round(3).values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code improvement column
        for i in range(1, len(df) + 1):
            improvement = df.iloc[i-1]['Improvement (%)']
            if improvement > 0:
                table[(i, 3)].set_facecolor('#90EE90')  # Light green
            elif improvement < 0:
                table[(i, 3)].set_facecolor('#FFB6C1')  # Light red
        
        # Style header
        for j in range(len(df.columns)):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Detailed Metrics Comparison\nPure RAG vs Graph-Aware RAG', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'detailed_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Recall comparison (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        k_values = [1, 3, 5, 10]
        pure_recalls = [self.pure_rag_metrics["recall_at_k"][str(k)] for k in k_values]
        graph_recalls = [self.graph_rag_metrics["recall_at_k"][str(k)] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        ax1.bar(x - width/2, pure_recalls, width, label='Pure RAG', alpha=0.8)
        ax1.bar(x + width/2, graph_recalls, width, label='Graph-Aware RAG', alpha=0.8)
        ax1.set_title('Recall@k Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'@{k}' for k in k_values])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Key metrics comparison (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        metrics = ['MRR', 'F1 Score', 'Exact Match']
        pure_values = [self.pure_rag_metrics["mrr"], self.pure_rag_metrics["f1_score"], self.pure_rag_metrics["exact_match"]]
        graph_values = [self.graph_rag_metrics["mrr"], self.graph_rag_metrics["f1_score"], self.graph_rag_metrics["exact_match"]]
        
        x = np.arange(len(metrics))
        ax2.bar(x - width/2, pure_values, width, label='Pure RAG', alpha=0.8)
        ax2.bar(x + width/2, graph_values, width, label='Graph-Aware RAG', alpha=0.8)
        ax2.set_title('Key Quality Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance times (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        times = ['Retrieval', 'Generation']
        pure_times = [self.pure_rag_metrics["avg_retrieval_time"], self.pure_rag_metrics["avg_generation_time"]]
        graph_times = [self.graph_rag_metrics["avg_retrieval_time"], self.graph_rag_metrics["avg_generation_time"]]
        
        x = np.arange(len(times))
        ax3.bar(x - width/2, pure_times, width, label='Pure RAG', alpha=0.8)
        ax3.bar(x + width/2, graph_times, width, label='Graph-Aware RAG', alpha=0.8)
        ax3.set_title('Processing Times (seconds)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(times)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Improvements (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        imp_metrics = ['recall_at_5', 'mrr', 'f1_score']
        improvements = [self.improvements.get(metric, 0) for metric in imp_metrics]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        bars = ax4.bar(range(len(imp_metrics)), improvements, color=colors, alpha=0.7)
        ax4.set_title('Improvements (%)')
        ax4.set_xticks(range(len(imp_metrics)))
        ax4.set_xticklabels(['Recall@5', 'MRR', 'F1 Score'])
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax4.annotate(f'{imp:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')
        
        # 5. System information (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create info text
        info_text = f"""
EVALUATION SUMMARY
Dataset: {self.results['dataset_info']['num_passages']} passages, {self.results['dataset_info']['num_queries']} queries
Evaluation Date: {self.results['evaluation_timestamp'][:10]}

PURE RAG SYSTEM:
• Method: FAISS Vector Search + LLM Generation
• Embedding Model: {self.results['pure_rag']['system_info'].get('embedding_model', 'N/A')}

GRAPH-AWARE RAG SYSTEM:
• Method: Semantic Graph + Vector Hybrid Retrieval
• Graph-Vector Weight (α): {self.results['graph_aware_rag']['system_info']['graph_vector_weight']}
• Expansion Depth: {self.results['graph_aware_rag']['system_info']['expansion_depth']}

KEY FINDINGS:
• Best Recall@5 Improvement: {self.improvements.get('recall_at_5', 0):.1f}%
• Best MRR Improvement: {self.improvements.get('mrr', 0):.1f}%
• Hallucination Rate Change: {self.improvements.get('hallucination_rate', 0):.1f}%
        """
        
        ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('RAG vs Graph-Aware Retrieval: Comprehensive Evaluation Results', 
                    fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization charts"""
        
        print("🎨 Generating evaluation visualizations...")
        
        try:
            self.create_recall_comparison()
            print("✅ Recall comparison chart created")
        except Exception as e:
            print(f"❌ Failed to create recall comparison: {e}")
        
        try:
            self.create_metrics_radar_chart()
            print("✅ Metrics radar chart created")
        except Exception as e:
            print(f"❌ Failed to create radar chart: {e}")
        
        try:
            self.create_improvement_chart()
            print("✅ Improvement chart created")
        except Exception as e:
            print(f"❌ Failed to create improvement chart: {e}")
        
        try:
            self.create_performance_comparison()
            print("✅ Performance comparison chart created")
        except Exception as e:
            print(f"❌ Failed to create performance chart: {e}")
        
        try:
            self.create_hyperparameter_tuning_heatmap()
            print("✅ Hyperparameter tuning heatmap created")
        except Exception as e:
            print(f"❌ Failed to create tuning heatmap: {e}")
        
        try:
            self.create_detailed_metrics_breakdown()
            print("✅ Detailed metrics breakdown created")
        except Exception as e:
            print(f"❌ Failed to create metrics breakdown: {e}")
        
        try:
            self.create_summary_dashboard()
            print("✅ Summary dashboard created")
        except Exception as e:
            print(f"❌ Failed to create summary dashboard: {e}")
        
        print(f"\n📊 All visualizations saved to: {self.plots_dir}")
        
        # List generated files
        plot_files = [f for f in os.listdir(self.plots_dir) if f.endswith('.png')]
        print(f"Generated {len(plot_files)} visualization files:")
        for file in sorted(plot_files):
            print(f"  • {file}")

def main():
    """Main visualization execution"""
    
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python evaluation_visualizer.py <results_json_path>")
        return
    
    results_path = sys.argv[1]
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    try:
        visualizer = EvaluationVisualizer(results_path)
        visualizer.generate_all_visualizations()
        print("\n🎉 Visualization generation completed!")
        
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
