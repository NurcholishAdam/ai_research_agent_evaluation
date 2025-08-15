#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG vs Graph-Aware Retrieval Evaluation Runner
Simplified script to run the complete evaluation and generate visualizations
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import faiss
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    try:
        import sentence_transformers
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import matplotlib
        import seaborn
    except ImportError:
        missing_deps.append("matplotlib seaborn")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        return False
    
    return True

def run_quick_evaluation():
    """Run a quick evaluation with smaller dataset"""
    
    print("🚀 Running Quick RAG vs Graph-Aware Evaluation")
    print("=" * 50)
    
    try:
        from evaluation.rag_vs_graph_evaluation import EvaluationFramework
        
        # Create evaluation framework
        evaluator = EvaluationFramework()
        
        # Run evaluation with smaller parameters for quick testing
        results = evaluator.run_full_evaluation(
            dataset_path="evaluation/quick_dataset",
            corpus_size=1000,    # Small corpus for quick testing
            num_queries=50,      # Small query set
            tune_hyperparameters=True
        )
        
        # Save results
        results_path = "evaluation/quick_evaluation_results.json"
        evaluator.generate_report(results, results_path)
        
        return results_path
        
    except Exception as e:
        print(f"❌ Quick evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_full_evaluation():
    """Run full evaluation with complete dataset"""
    
    print("🚀 Running Full RAG vs Graph-Aware Evaluation")
    print("=" * 50)
    
    try:
        from evaluation.rag_vs_graph_evaluation import EvaluationFramework
        
        # Create evaluation framework
        evaluator = EvaluationFramework()
        
        # Run full evaluation
        results = evaluator.run_full_evaluation(
            dataset_path="evaluation/full_dataset",
            corpus_size=10000,   # Full corpus
            num_queries=500,     # Full query set
            tune_hyperparameters=True
        )
        
        # Save results
        results_path = "evaluation/full_evaluation_results.json"
        evaluator.generate_report(results, results_path)
        
        return results_path
        
    except Exception as e:
        print(f"❌ Full evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_visualizations(results_path):
    """Generate visualizations from results"""
    
    if not results_path or not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return False
    
    try:
        from evaluation.evaluation_visualizer import EvaluationVisualizer
        
        print(f"\n🎨 Generating visualizations from {results_path}")
        
        visualizer = EvaluationVisualizer(results_path)
        visualizer.generate_all_visualizations()
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_demo_evaluation():
    """Run a minimal demo evaluation for testing"""
    
    print("🧪 Running Demo RAG vs Graph-Aware Evaluation")
    print("=" * 50)
    
    try:
        from evaluation.rag_vs_graph_evaluation import EvaluationFramework
        
        # Create evaluation framework
        evaluator = EvaluationFramework()
        
        # Run minimal evaluation for demo
        results = evaluator.run_full_evaluation(
            dataset_path="evaluation/demo_dataset",
            corpus_size=100,     # Minimal corpus
            num_queries=10,      # Minimal queries
            tune_hyperparameters=False  # Skip tuning for speed
        )
        
        # Save results
        results_path = "evaluation/demo_evaluation_results.json"
        evaluator.generate_report(results, results_path)
        
        return results_path
        
    except Exception as e:
        print(f"❌ Demo evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_usage():
    """Print usage information"""
    
    print("""
🔬 RAG vs Graph-Aware Retrieval Evaluation

Usage:
    python run_evaluation.py [mode]

Modes:
    demo    - Quick demo with minimal dataset (100 passages, 10 queries)
    quick   - Quick evaluation (1K passages, 50 queries)  
    full    - Full evaluation (10K passages, 500 queries)
    viz     - Generate visualizations from existing results

Examples:
    python run_evaluation.py demo
    python run_evaluation.py quick
    python run_evaluation.py full
    python run_evaluation.py viz evaluation/results.json

Requirements:
    pip install faiss-cpu sentence-transformers matplotlib seaborn pandas numpy
    """)

def main():
    """Main execution function"""
    
    # Check dependencies first
    if not check_dependencies():
        print("\n💡 Install missing dependencies and try again")
        return
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='RAG vs Graph-Aware Retrieval Evaluation')
    parser.add_argument('mode', choices=['demo', 'quick', 'full', 'viz'], 
                       help='Evaluation mode to run')
    parser.add_argument('--results-path', type=str, 
                       help='Path to results file (for viz mode)')
    
    args = parser.parse_args()
    
    # Create evaluation directory
    os.makedirs('evaluation', exist_ok=True)
    
    start_time = datetime.now()
    
    try:
        if args.mode == 'demo':
            results_path = run_demo_evaluation()
            
        elif args.mode == 'quick':
            results_path = run_quick_evaluation()
            
        elif args.mode == 'full':
            results_path = run_full_evaluation()
            
        elif args.mode == 'viz':
            if not args.results_path:
                print("❌ --results-path required for viz mode")
                return
            results_path = args.results_path
        
        # Generate visualizations if we have results
        if results_path:
            print(f"\n📊 Results saved to: {results_path}")
            
            # Generate visualizations
            viz_success = generate_visualizations(results_path)
            
            if viz_success:
                plots_dir = os.path.join(os.path.dirname(results_path), "plots")
                print(f"📈 Visualizations saved to: {plots_dir}")
                
                # List generated plots
                if os.path.exists(plots_dir):
                    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                    print(f"\nGenerated {len(plot_files)} visualization files:")
                    for file in sorted(plot_files):
                        print(f"  • {file}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 Evaluation completed in {duration:.1f} seconds!")
        
        # Print summary
        if results_path and os.path.exists(results_path):
            try:
                import json
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                print(f"\n📈 QUICK SUMMARY:")
                pure_metrics = results["pure_rag"]["metrics"]
                graph_metrics = results["graph_aware_rag"]["metrics"]
                improvements = results["improvements"]
                
                print(f"Recall@5:     Pure RAG: {pure_metrics['recall_at_k']['5']:.3f} | Graph-Aware: {graph_metrics['recall_at_k']['5']:.3f} | Improvement: {improvements.get('recall_at_5', 0):.1f}%")
                print(f"MRR:          Pure RAG: {pure_metrics['mrr']:.3f} | Graph-Aware: {graph_metrics['mrr']:.3f} | Improvement: {improvements.get('mrr', 0):.1f}%")
                print(f"F1 Score:     Pure RAG: {pure_metrics['f1_score']:.3f} | Graph-Aware: {graph_metrics['f1_score']:.3f} | Improvement: {improvements.get('f1_score', 0):.1f}%")
                
            except Exception as e:
                print(f"Could not load summary: {e}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()
    else:
        main()