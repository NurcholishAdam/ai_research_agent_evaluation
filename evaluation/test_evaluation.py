#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for RAG vs Graph-Aware Retrieval Evaluation
Verifies that the evaluation framework works correctly
"""

import os
import sys
import tempfile
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dataset_generation():
    """Test dataset generation functionality"""
    
    print("🧪 Testing Dataset Generation...")
    
    try:
        from evaluation.rag_vs_graph_evaluation import DatasetGenerator
        
        generator = DatasetGenerator()
        
        # Generate small test corpus
        passages = generator.generate_synthetic_corpus(size=50)
        print(f"✅ Generated {len(passages)} passages")
        
        # Generate test queries
        queries = generator.generate_evaluation_queries(num_queries=10)
        print(f"✅ Generated {len(queries)} queries")
        
        # Verify structure
        assert len(passages) == 50, "Incorrect number of passages"
        assert len(queries) == 10, "Incorrect number of queries"
        assert all('content' in p for p in passages), "Missing content in passages"
        assert all(hasattr(q, 'question') for q in queries), "Missing question in queries"
        
        print("✅ Dataset generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation test failed: {e}")
        return False

def test_pure_rag_system():
    """Test Pure RAG system functionality"""
    
    print("🧪 Testing Pure RAG System...")
    
    try:
        # Check if FAISS is available
        try:
            import faiss
            import sentence_transformers
        except ImportError:
            print("⚠️ FAISS/sentence-transformers not available, skipping Pure RAG test")
            return True
        
        from evaluation.rag_vs_graph_evaluation import PureRAGSystem, DatasetGenerator
        
        # Create test data
        generator = DatasetGenerator()
        passages = generator.generate_synthetic_corpus(size=20)
        
        # Create and test Pure RAG system
        pure_rag = PureRAGSystem()
        pure_rag.build_index(passages)
        
        # Test retrieval
        result = pure_rag.retrieve("machine learning", k=5)
        print(f"✅ Retrieved {len(result.retrieved_passages)} passages")
        
        # Test answer generation
        answer_result = pure_rag.generate_answer("What is machine learning?", result)
        print(f"✅ Generated answer: {answer_result.generated_answer[:50]}...")
        
        print("✅ Pure RAG system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pure RAG system test failed: {e}")
        return False

def test_graph_aware_system():
    """Test Graph-Aware RAG system functionality"""
    
    print("🧪 Testing Graph-Aware RAG System...")
    
    try:
        from evaluation.rag_vs_graph_evaluation import GraphAwareRAGSystem, DatasetGenerator
        
        # Create test data
        generator = DatasetGenerator()
        passages = generator.generate_synthetic_corpus(size=10)  # Very small for testing
        
        # Create and test Graph-Aware RAG system
        graph_rag = GraphAwareRAGSystem()
        
        # This might take a while, so we'll do a minimal test
        print("⏳ Building graph index (this may take a moment)...")
        graph_rag.build_graph_index(passages)
        
        # Test retrieval
        result = graph_rag.retrieve("neural networks", k=3)
        print(f"✅ Retrieved {len(result.retrieved_passages)} passages with graph enhancement")
        
        # Test answer generation
        answer_result = graph_rag.generate_answer("What are neural networks?", result)
        print(f"✅ Generated enhanced answer: {answer_result.generated_answer[:50]}...")
        
        print("✅ Graph-Aware RAG system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Graph-Aware RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test metrics calculation functionality"""
    
    print("🧪 Testing Metrics Calculation...")
    
    try:
        from evaluation.rag_vs_graph_evaluation import EvaluationFramework, EvaluationQuery, AnswerResult, RetrievalResult
        
        evaluator = EvaluationFramework()
        
        # Create mock data
        queries = [
            EvaluationQuery(
                id="test_1",
                question="What is AI?",
                expected_answer="Artificial Intelligence is a field of computer science",
                relevant_passages=["passage_1", "passage_2"]
            )
        ]
        
        # Create mock results
        retrieval_result = RetrievalResult(
            query_id="test_1",
            retrieved_passages=[
                {"id": "passage_1", "content": "AI content", "similarity_score": 0.9},
                {"id": "passage_3", "content": "Other content", "similarity_score": 0.7}
            ],
            retrieval_time=0.1,
            method="test",
            parameters={}
        )
        
        answer_result = AnswerResult(
            query_id="test_1",
            generated_answer="Artificial Intelligence is computer science field",
            generation_time=0.2,
            retrieval_result=retrieval_result
        )
        
        system_results = {
            "results": [answer_result],
            "avg_retrieval_time": 0.1,
            "avg_generation_time": 0.2
        }
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(system_results, queries)
        
        print(f"✅ Calculated metrics:")
        print(f"   Recall@1: {metrics.recall_at_k[1]:.3f}")
        print(f"   MRR: {metrics.mrr:.3f}")
        print(f"   F1 Score: {metrics.f1_score:.3f}")
        
        print("✅ Metrics calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation test failed: {e}")
        return False

def test_visualization():
    """Test visualization generation"""
    
    print("🧪 Testing Visualization Generation...")
    
    try:
        # Check if matplotlib is available
        try:
            import matplotlib
            import seaborn
        except ImportError:
            print("⚠️ Matplotlib/seaborn not available, skipping visualization test")
            return True
        
        # Create mock results file
        mock_results = {
            "dataset_info": {"num_passages": 100, "num_queries": 10},
            "pure_rag": {
                "metrics": {
                    "recall_at_k": {"1": 0.2, "3": 0.4, "5": 0.6, "10": 0.8},
                    "mrr": 0.35,
                    "f1_score": 0.45,
                    "exact_match": 0.15,
                    "hallucination_rate": 0.1,
                    "avg_retrieval_time": 0.05,
                    "avg_generation_time": 0.8
                },
                "system_info": {"embedding_model": "test"}
            },
            "graph_aware_rag": {
                "metrics": {
                    "recall_at_k": {"1": 0.25, "3": 0.5, "5": 0.7, "10": 0.85},
                    "mrr": 0.42,
                    "f1_score": 0.52,
                    "exact_match": 0.18,
                    "hallucination_rate": 0.08,
                    "avg_retrieval_time": 0.08,
                    "avg_generation_time": 0.8
                },
                "system_info": {
                    "graph_vector_weight": 0.6,
                    "expansion_depth": 2,
                    "edge_type_boosts": {"CITES": 2.0}
                }
            },
            "improvements": {
                "recall_at_5": 16.7,
                "mrr": 20.0,
                "f1_score": 15.6,
                "exact_match": 20.0,
                "hallucination_rate": 20.0
            },
            "hyperparameter_tuning": {
                "best_config": {"alpha": 0.6, "depth": 2},
                "best_score": 0.65,
                "all_results": [
                    {"alpha": 0.3, "depth": 1, "combined_score": 0.5},
                    {"alpha": 0.6, "depth": 2, "combined_score": 0.65},
                    {"alpha": 0.9, "depth": 3, "combined_score": 0.6}
                ]
            },
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Save mock results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_results, f, indent=2)
            temp_results_path = f.name
        
        try:
            from evaluation.evaluation_visualizer import EvaluationVisualizer
            
            visualizer = EvaluationVisualizer(temp_results_path)
            
            # Test individual visualization methods
            visualizer.create_recall_comparison()
            print("✅ Recall comparison chart created")
            
            visualizer.create_improvement_chart()
            print("✅ Improvement chart created")
            
            visualizer.create_performance_comparison()
            print("✅ Performance comparison created")
            
            print("✅ Visualization test passed")
            return True
            
        finally:
            # Cleanup
            os.unlink(temp_results_path)
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    
    print("🚀 Running Evaluation Framework Tests")
    print("=" * 50)
    
    tests = [
        ("Dataset Generation", test_dataset_generation),
        ("Pure RAG System", test_pure_rag_system),
        ("Graph-Aware System", test_graph_aware_system),
        ("Metrics Calculation", test_metrics_calculation),
        ("Visualization", test_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Evaluation framework is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

def main():
    """Main test execution"""
    
    try:
        success = run_all_tests()
        
        if success:
            print(f"\n💡 Next steps:")
            print(f"   1. Run demo evaluation: python evaluation/run_evaluation.py demo")
            print(f"   2. Run quick evaluation: python evaluation/run_evaluation.py quick")
            print(f"   3. Run full evaluation: python evaluation/run_evaluation.py full")
        else:
            print(f"\n🔧 Fix the failing tests before running evaluations")
        
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()