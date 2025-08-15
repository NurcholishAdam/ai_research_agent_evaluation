# 🔬 RAG vs Graph-Aware Retrieval Evaluation Framework

This comprehensive evaluation framework compares **Pure RAG** (Retrieval-Augmented Generation) against our **Graph-Aware Retrieval** system using rigorous experimental methodology.

## 📋 Overview

The evaluation implements a controlled comparison between:

1. **Pure RAG System**: FAISS vector search + LLM generation
2. **Graph-Aware RAG System**: Semantic graph + vector hybrid retrieval + LLM generation

## 🎯 Experimental Setup

### Dataset Generation
- **Corpus Size**: 10K–50K synthetic passages
- **Domains**: Machine Learning, NLP, Computer Vision, Data Science
- **Query Set**: 500 evaluation queries with ground truth
- **Ground Truth**: Question-passage mappings with relevance annotations

### Systems Under Test

#### Pure RAG System
- **Vector Retrieval**: FAISS with sentence-transformers embeddings
- **Top-k Selection**: Cosine similarity ranking
- **Answer Generation**: Groq LLM with retrieved context

#### Graph-Aware RAG System
- **Initial Retrieval**: Same vector model as Pure RAG
- **Graph Enhancement**: Reranking and expansion using `GraphAwareRetrieval.retrieve()`
- **Path-based Boosting**: Neighborhood expansion with graph traversal
- **Answer Generation**: Same LLM with enhanced context

### Key Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Recall@k** | Fraction of queries with relevant passage in top-k | `relevant_in_topk / total_queries` |
| **MRR** | Mean Reciprocal Rank of first relevant passage | `mean(1/rank_first_relevant)` |
| **Exact Match** | Percentage of exact answer matches | `exact_matches / total_queries` |
| **F1 Score** | Token overlap between generated and reference | `2 * precision * recall / (precision + recall)` |
| **Hallucination Rate** | Percentage of unsupported statements | `hallucinated_answers / total_queries` |

### Hyperparameter Tuning

#### Graph vs Vector Weight (α)
Blends similarity scores: `score_combined = α × score_graph + (1-α) × score_vector`
- **Range**: α ∈ [0.1, 0.9]
- **Purpose**: Balance local graph context against global embeddings

#### Expansion Depth (h)
Number of graph hops for neighbor inclusion
- **Range**: h = 1, 2, 3
- **Trade-off**: Context diversity vs noise

#### Edge-Type Boosts
Custom weights for different edge types:
- **CITES**: +2.0× (authoritative connections)
- **MENTIONS**: +1.5× (content relationships)
- **USES**: +1.8× (tool/method relationships)

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install faiss-cpu sentence-transformers matplotlib seaborn pandas numpy

# Install project dependencies
pip install -r requirements.txt
```

### Running Evaluations

#### Demo Evaluation (Quick Test)
```bash
python evaluation/run_evaluation.py demo
```
- **Dataset**: 100 passages, 10 queries
- **Duration**: ~2-3 minutes
- **Purpose**: Test setup and basic functionality

#### Quick Evaluation
```bash
python evaluation/run_evaluation.py quick
```
- **Dataset**: 1K passages, 50 queries
- **Duration**: ~10-15 minutes
- **Purpose**: Rapid comparison with meaningful results

#### Full Evaluation
```bash
python evaluation/run_evaluation.py full
```
- **Dataset**: 10K passages, 500 queries
- **Duration**: ~1-2 hours
- **Purpose**: Comprehensive comparison with statistical significance

#### Generate Visualizations Only
```bash
python evaluation/run_evaluation.py viz evaluation/results.json
```

### Manual Execution

```python
from evaluation.rag_vs_graph_evaluation import EvaluationFramework

# Create framework
evaluator = EvaluationFramework()

# Run evaluation
results = evaluator.run_full_evaluation(
    corpus_size=5000,
    num_queries=200,
    tune_hyperparameters=True
)

# Generate report
evaluator.generate_report(results, "my_results.json")
```

## 📊 Output Files

### Results JSON
```json
{
  "dataset_info": {
    "num_passages": 10000,
    "num_queries": 500
  },
  "pure_rag": {
    "metrics": {
      "recall_at_k": {"1": 0.234, "5": 0.567, "10": 0.789},
      "mrr": 0.345,
      "f1_score": 0.456,
      "exact_match": 0.123,
      "hallucination_rate": 0.089
    }
  },
  "graph_aware_rag": {
    "metrics": { /* same structure */ }
  },
  "improvements": {
    "recall_at_5": 23.4,
    "mrr": 15.6,
    "f1_score": 12.3
  },
  "hyperparameter_tuning": {
    "best_config": {"alpha": 0.6, "depth": 2},
    "best_score": 0.678
  }
}
```

### Generated Visualizations

1. **`recall_comparison.png`** - Recall@k bar chart comparison
2. **`metrics_radar.png`** - Multi-metric radar chart
3. **`improvements.png`** - Improvement percentages
4. **`performance_comparison.png`** - Processing time comparison
5. **`hyperparameter_tuning.png`** - Tuning results heatmap
6. **`detailed_metrics.png`** - Comprehensive metrics table
7. **`summary_dashboard.png`** - Complete evaluation dashboard

## 🔧 Configuration Options

### Dataset Configuration
```python
# Custom dataset generation
generator = DatasetGenerator()
passages = generator.generate_synthetic_corpus(
    size=15000,
    domains=["machine_learning", "nlp", "computer_vision"]
)
queries = generator.generate_evaluation_queries(
    num_queries=750,
    difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2}
)
```

### System Configuration
```python
# Pure RAG system
pure_rag = PureRAGSystem(embedding_model="all-MiniLM-L6-v2")

# Graph-aware system with custom parameters
graph_rag = GraphAwareRAGSystem()
graph_rag.tune_hyperparameters(
    graph_vector_weight=0.7,
    expansion_depth=3,
    edge_type_boosts={"CITES": 2.5, "MENTIONS": 1.8}
)
```

### Evaluation Configuration
```python
evaluator = EvaluationFramework()
results = evaluator.run_full_evaluation(
    dataset_path="custom/dataset",
    corpus_size=20000,
    num_queries=1000,
    tune_hyperparameters=True,
    embedding_model="all-mpnet-base-v2"  # Higher quality embeddings
)
```

## 📈 Expected Results

Based on our semantic graph architecture, we expect to see:

### Retrieval Improvements
- **Recall@5**: +15-25% improvement through graph expansion
- **MRR**: +10-20% improvement through better ranking
- **Context Quality**: Enhanced through relationship-aware retrieval

### Answer Quality Improvements
- **F1 Score**: +8-15% improvement through better context
- **Exact Match**: +5-12% improvement through precise retrieval
- **Hallucination Rate**: -10-20% reduction through grounded context

### Performance Trade-offs
- **Retrieval Time**: +20-50% increase due to graph processing
- **Generation Time**: Similar (same LLM)
- **Memory Usage**: +30-60% increase for graph storage

## 🔍 Analysis Features

### Hyperparameter Analysis
- **Grid Search**: Systematic exploration of α and depth parameters
- **Performance Surface**: Visualization of parameter impact
- **Optimal Configuration**: Automatic selection of best parameters

### Error Analysis
- **Query Difficulty**: Performance breakdown by query complexity
- **Failure Cases**: Analysis of queries where graph-aware performs worse
- **Domain Analysis**: Performance by research domain

### Statistical Significance
- **Confidence Intervals**: Bootstrap confidence intervals for metrics
- **Significance Tests**: Statistical tests for improvement claims
- **Effect Sizes**: Practical significance of improvements

## 🛠️ Extending the Framework

### Adding New Metrics
```python
def custom_metric(generated_answer, reference_answer, retrieved_passages):
    # Implement custom evaluation logic
    return score

# Add to evaluation framework
evaluator.add_custom_metric("custom_metric", custom_metric)
```

### Adding New Systems
```python
class CustomRAGSystem:
    def retrieve(self, query, k=10):
        # Implement custom retrieval
        pass
    
    def generate_answer(self, query, retrieval_result):
        # Implement custom generation
        pass

# Add to comparison
evaluator.add_system("custom_rag", CustomRAGSystem())
```

### Custom Datasets
```python
# Load external dataset
def load_natural_questions():
    # Load Natural Questions dataset
    return passages, queries

# Use in evaluation
passages, queries = load_natural_questions()
evaluator.evaluate_on_dataset(passages, queries)
```

## 📚 Research Applications

This framework can be used for:

1. **Academic Research**: Rigorous comparison of retrieval methods
2. **System Development**: Iterative improvement of graph-aware systems
3. **Hyperparameter Optimization**: Finding optimal configurations
4. **Ablation Studies**: Understanding component contributions
5. **Benchmark Creation**: Establishing performance baselines

## 🤝 Contributing

To contribute to the evaluation framework:

1. **Add New Metrics**: Implement additional evaluation metrics
2. **Extend Datasets**: Add support for real-world datasets
3. **Improve Visualizations**: Create new analysis charts
4. **Optimize Performance**: Improve evaluation speed
5. **Add Documentation**: Enhance usage examples

## 📄 Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{rag_graph_evaluation,
  title={RAG vs Graph-Aware Retrieval Evaluation Framework},
  author={AI Research Agent Team},
  year={2024},
  url={https://github.com/your-repo/evaluation}
}
```

## 🔗 Related Work

- **RAG**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Graph Neural Networks**: Learning on Graphs with Deep Learning
- **Semantic Search**: Beyond Vector Similarity in Information Retrieval
- **Knowledge Graphs**: Structured Knowledge for AI Systems

---

For questions or issues, please open a GitHub issue or contact the development team.