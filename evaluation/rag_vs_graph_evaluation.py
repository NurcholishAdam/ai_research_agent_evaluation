#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG vs Graph-Aware Retrieval Evaluation Framework
Comprehensive comparison of pure RAG against semantic graph-enhanced retrieval
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import time
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Vector search and embeddings
try:
    import faiss
    import sentence_transformers
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS and sentence-transformers not available. Install with: pip install faiss-cpu sentence-transformers")

# Our semantic graph components
from semantic_graph.research_agent_integration import create_semantic_graph_agent
from semantic_graph.graph_retrieval import RetrievalStrategy
from semantic_graph.graph_core import NodeType

# LLM for answer generation
from llm.groq_wrapper import load_groq_llm

logger = logging.getLogger(__name__)

@dataclass
class EvaluationQuery:
    """Represents a single evaluation query"""
    id: str
    question: str
    expected_answer: str
    relevant_passages: List[str]
    context: str = ""
    difficulty: str = "medium"  # easy, medium, hard

@dataclass
class RetrievalResult:
    """Represents retrieval results for a query"""
    query_id: str
    retrieved_passages: List[Dict[str, Any]]
    retrieval_time: float
    method: str
    parameters: Dict[str, Any]

@dataclass
class AnswerResult:
    """Represents answer generation results"""
    query_id: str
    generated_answer: str
    generation_time: float
    retrieval_result: RetrievalResult
    confidence_score: float = 0.0

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    recall_at_k: Dict[int, float]
    mrr: float
    exact_match: float
    f1_score: float
    hallucination_rate: float
    avg_retrieval_time: float
    avg_generation_time: float
    total_queries: int

class DatasetGenerator:
    """Generates evaluation datasets for RAG vs Graph comparison"""
    
    def __init__(self):
        self.passages = []
        self.queries = []
        
    def generate_synthetic_corpus(self, size: int = 10000) -> List[Dict[str, Any]]:
        """Generate synthetic corpus of passages for evaluation"""
        
        # Research domains and topics
        domains = {
            "machine_learning": [
                "neural networks", "deep learning", "transformers", "attention mechanisms",
                "convolutional networks", "recurrent networks", "reinforcement learning",
                "supervised learning", "unsupervised learning", "transfer learning"
            ],
            "natural_language_processing": [
                "language models", "text classification", "named entity recognition",
                "sentiment analysis", "machine translation", "question answering",
                "text summarization", "information extraction", "dialogue systems"
            ],
            "computer_vision": [
                "image classification", "object detection", "semantic segmentation",
                "face recognition", "optical character recognition", "image generation",
                "video analysis", "3D reconstruction", "medical imaging"
            ],
            "data_science": [
                "data mining", "statistical analysis", "predictive modeling",
                "feature engineering", "dimensionality reduction", "clustering",
                "time series analysis", "A/B testing", "data visualization"
            ]
        }
        
        passages = []
        passage_id = 0
        
        for domain, topics in domains.items():
            for topic in topics:
                # Generate multiple passages per topic
                for i in range(size // (len(domains) * len(topics))):
                    passage = self._generate_passage(domain, topic, passage_id)
                    passages.append(passage)
                    passage_id += 1
        
        # Fill remaining slots with random combinations
        while len(passages) < size:
            domain = np.random.choice(list(domains.keys()))
            topic = np.random.choice(domains[domain])
            passage = self._generate_passage(domain, topic, passage_id)
            passages.append(passage)
            passage_id += 1
        
        self.passages = passages
        return passages
    
    def _generate_passage(self, domain: str, topic: str, passage_id: int) -> Dict[str, Any]:
        """Generate a single passage with metadata"""
        
        # Template-based passage generation
        templates = [
            f"{topic.title()} is a fundamental concept in {domain.replace('_', ' ')}. "
            f"Recent advances in {topic} have shown significant improvements in performance. "
            f"Key applications include automated systems and intelligent processing. "
            f"Researchers have developed novel approaches that leverage {topic} for enhanced results.",
            
            f"In the field of {domain.replace('_', ' ')}, {topic} plays a crucial role. "
            f"Modern implementations of {topic} utilize advanced algorithms and techniques. "
            f"The methodology involves systematic approaches to problem-solving. "
            f"Experimental results demonstrate the effectiveness of {topic}-based solutions.",
            
            f"The study of {topic} within {domain.replace('_', ' ')} has evolved significantly. "
            f"Current research focuses on improving {topic} through innovative methods. "
            f"Applications span across various industries and use cases. "
            f"Future developments in {topic} promise even greater capabilities."
        ]
        
        content = np.random.choice(templates)
        
        # Add some citations and technical details
        citations = [
            f"Smith et al. (2023)", f"Johnson and Lee (2022)", f"Brown et al. (2024)",
            f"Davis and Wilson (2023)", f"Chen et al. (2022)"
        ]
        
        content += f" Studies by {np.random.choice(citations)} provide empirical evidence. "
        content += f"The {topic} approach achieves {np.random.randint(85, 98)}% accuracy on benchmark datasets."
        
        return {
            "id": f"passage_{passage_id}",
            "content": content,
            "domain": domain,
            "topic": topic,
            "length": len(content.split()),
            "citations": [np.random.choice(citations)],
            "concepts": [topic, domain.replace('_', ' ')],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source": "synthetic_generation",
                "quality_score": np.random.uniform(0.7, 1.0)
            }
        }
    
    def generate_evaluation_queries(self, num_queries: int = 500) -> List[EvaluationQuery]:
        """Generate evaluation queries with ground truth"""
        
        if not self.passages:
            raise ValueError("Generate corpus first using generate_synthetic_corpus()")
        
        queries = []
        
        # Query templates
        question_templates = [
            "What is {topic}?",
            "How does {topic} work?",
            "What are the applications of {topic}?",
            "What are recent advances in {topic}?",
            "How is {topic} used in {domain}?",
            "What are the benefits of {topic}?",
            "What challenges exist in {topic}?",
            "How can {topic} be improved?",
            "What is the future of {topic}?",
            "How does {topic} compare to other approaches?"
        ]
        
        for i in range(num_queries):
            # Select random passage as ground truth
            relevant_passage = np.random.choice(self.passages)
            topic = relevant_passage["topic"]
            domain = relevant_passage["domain"]
            
            # Generate question
            template = np.random.choice(question_templates)
            question = template.format(topic=topic, domain=domain.replace('_', ' '))
            
            # Generate expected answer based on passage content
            expected_answer = self._generate_expected_answer(relevant_passage, question)
            
            # Find other relevant passages (for recall evaluation)
            relevant_passages = [relevant_passage["id"]]
            for passage in self.passages:
                if (passage["topic"] == topic or passage["domain"] == domain) and passage["id"] != relevant_passage["id"]:
                    if len(relevant_passages) < 5:  # Limit to 5 relevant passages
                        relevant_passages.append(passage["id"])
            
            # Determine difficulty
            difficulty = self._determine_difficulty(question, relevant_passages)
            
            query = EvaluationQuery(
                id=f"query_{i}",
                question=question,
                expected_answer=expected_answer,
                relevant_passages=relevant_passages,
                context=f"Domain: {domain}, Topic: {topic}",
                difficulty=difficulty
            )
            
            queries.append(query)
        
        self.queries = queries
        return queries
    
    def _generate_expected_answer(self, passage: Dict[str, Any], question: str) -> str:
        """Generate expected answer based on passage content and question"""
        
        content = passage["content"]
        topic = passage["topic"]
        domain = passage["domain"]
        
        if "what is" in question.lower():
            return f"{topic.title()} is a key concept in {domain.replace('_', ' ')} that involves advanced techniques and methodologies."
        elif "how does" in question.lower():
            return f"{topic.title()} works through systematic approaches and algorithms that process data effectively."
        elif "applications" in question.lower():
            return f"{topic.title()} has applications in automated systems, intelligent processing, and various industry use cases."
        elif "recent advances" in question.lower():
            return f"Recent advances in {topic} include improved algorithms, better performance metrics, and novel methodological approaches."
        else:
            # Extract key sentence from passage
            sentences = content.split('. ')
            return sentences[0] + "."
    
    def _determine_difficulty(self, question: str, relevant_passages: List[str]) -> str:
        """Determine query difficulty based on question complexity and relevant passages"""
        
        if len(relevant_passages) <= 2:
            return "hard"
        elif len(relevant_passages) <= 4:
            return "medium"
        else:
            return "easy"
    
    def save_dataset(self, output_dir: str):
        """Save generated dataset to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save passages
        with open(os.path.join(output_dir, "passages.json"), "w") as f:
            json.dump(self.passages, f, indent=2)
        
        # Save queries
        queries_data = [asdict(query) for query in self.queries]
        with open(os.path.join(output_dir, "queries.json"), "w") as f:
            json.dump(queries_data, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Passages: {len(self.passages)}")
        print(f"Queries: {len(self.queries)}")

class PureRAGSystem:
    """Pure RAG system using FAISS vector search"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu sentence-transformers")
        
        self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model)
        self.index = None
        self.passages = []
        self.passage_embeddings = None
        self.llm = load_groq_llm()
        
    def build_index(self, passages: List[Dict[str, Any]]):
        """Build FAISS index from passages"""
        
        self.passages = passages
        
        # Extract text content
        texts = [passage["content"] for passage in passages]
        
        # Generate embeddings
        print("Generating embeddings for passages...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.passage_embeddings = embeddings
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Built FAISS index with {len(passages)} passages")
    
    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Retrieve top-k passages using vector similarity"""
        
        start_time = time.time()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Format results
        retrieved_passages = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.passages):
                passage = self.passages[idx].copy()
                passage["similarity_score"] = float(score)
                passage["rank"] = i + 1
                retrieved_passages.append(passage)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            query_id="",  # Will be set by caller
            retrieved_passages=retrieved_passages,
            retrieval_time=retrieval_time,
            method="pure_rag",
            parameters={"k": k, "embedding_model": self.embedding_model.get_sentence_embedding_dimension()}
        )
    
    def generate_answer(self, query: str, retrieval_result: RetrievalResult) -> AnswerResult:
        """Generate answer using retrieved passages"""
        
        start_time = time.time()
        
        # Prepare context from retrieved passages
        context_passages = []
        for passage in retrieval_result.retrieved_passages[:5]:  # Use top 5
            context_passages.append(f"Passage: {passage['content']}")
        
        context = "\n\n".join(context_passages)
        
        # Generate answer
        prompt = f"""Based on the following passages, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            generated_answer = response.content.strip()
        except Exception as e:
            generated_answer = f"Error generating answer: {str(e)}"
        
        generation_time = time.time() - start_time
        
        return AnswerResult(
            query_id=retrieval_result.query_id,
            generated_answer=generated_answer,
            generation_time=generation_time,
            retrieval_result=retrieval_result
        )

class GraphAwareRAGSystem:
    """Graph-aware RAG system using semantic graph enhancement"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.semantic_graph_agent = create_semantic_graph_agent()
        self.embedding_model_name = embedding_model
        self.llm = load_groq_llm()
        self.passages_indexed = False
        
        # Hyperparameters for tuning
        self.graph_vector_weight = 0.6  # α in the formula
        self.expansion_depth = 2
        self.edge_type_boosts = {
            "CITES": 2.0,
            "MENTIONS": 1.5,
            "RELATED_TO": 1.0,
            "USES": 1.8,
            "IMPLEMENTS": 1.6
        }
    
    def build_graph_index(self, passages: List[Dict[str, Any]]):
        """Build semantic graph from passages"""
        
        print("Building semantic graph from passages...")
        
        # Initialize graph with training data
        training_contexts = [passage["content"] for passage in passages[:100]]  # Sample for initialization
        self.semantic_graph_agent.initialize_with_training_data(training_contexts)
        
        # Ingest all passages into the graph
        for i, passage in enumerate(passages):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(passages)} passages")
            
            # Ingest passage as memory
            self.semantic_graph_agent.on_memory_write({
                'content': passage["content"],
                'importance': passage["metadata"]["quality_score"],
                'concepts': passage["concepts"],
                'citations': passage.get("citations", []),
                'memory_id': passage["id"],
                'domain': passage["domain"],
                'topic': passage["topic"]
            })
        
        # Process all ingestion events
        self.semantic_graph_agent.process_pending_ingestion()
        
        self.passages_indexed = True
        print(f"Built semantic graph with passages")
        
        # Get graph statistics
        stats = self.semantic_graph_agent.get_comprehensive_stats()
        print(f"Graph nodes: {stats['graph_core']['total_nodes']}")
        print(f"Graph edges: {stats['graph_core']['total_edges']}")
    
    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Retrieve using graph-aware methods"""
        
        if not self.passages_indexed:
            raise ValueError("Build graph index first using build_graph_index()")
        
        start_time = time.time()
        
        # Use graph-aware retrieval with current hyperparameters
        results = self.semantic_graph_agent.enhanced_retrieval(
            query=query,
            strategy="hybrid",  # Combines vector and graph
            top_k=k,
            node_types=["concept", "finding", "paper", "method"]
        )
        
        # Format results to match PureRAG format
        retrieved_passages = []
        for i, result in enumerate(results.get("results", [])):
            # Convert graph result to passage format
            passage_data = {
                "id": result["node_id"],
                "content": result["properties"].get("content", result["label"]),
                "similarity_score": result["score"],
                "rank": i + 1,
                "retrieval_method": result["method"],
                "explanation": result["explanation"],
                "graph_path": result.get("graph_path", []),
                "neighborhood_boost": result.get("neighborhood_boost", 0.0)
            }
            retrieved_passages.append(passage_data)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            query_id="",  # Will be set by caller
            retrieved_passages=retrieved_passages,
            retrieval_time=retrieval_time,
            method="graph_aware",
            parameters={
                "k": k,
                "graph_vector_weight": self.graph_vector_weight,
                "expansion_depth": self.expansion_depth,
                "edge_type_boosts": self.edge_type_boosts
            }
        )
    
    def generate_answer(self, query: str, retrieval_result: RetrievalResult) -> AnswerResult:
        """Generate answer using graph-enhanced retrieved passages"""
        
        start_time = time.time()
        
        # Prepare enhanced context from graph-retrieved passages
        context_passages = []
        for passage in retrieval_result.retrieved_passages[:5]:  # Use top 5
            context_text = f"Passage (Score: {passage['similarity_score']:.3f}): {passage['content']}"
            
            # Add graph-specific information
            if passage.get("explanation"):
                context_text += f"\nRetrieval Explanation: {passage['explanation']}"
            
            if passage.get("graph_path"):
                context_text += f"\nGraph Path: {' -> '.join(passage['graph_path'])}"
            
            context_passages.append(context_text)
        
        context = "\n\n".join(context_passages)
        
        # Enhanced prompt with graph information
        prompt = f"""Based on the following graph-enhanced passages, answer the question accurately and concisely. 
The passages have been retrieved using semantic graph relationships and may include explanations of why they are relevant.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            generated_answer = response.content.strip()
        except Exception as e:
            generated_answer = f"Error generating answer: {str(e)}"
        
        generation_time = time.time() - start_time
        
        return AnswerResult(
            query_id=retrieval_result.query_id,
            generated_answer=generated_answer,
            generation_time=generation_time,
            retrieval_result=retrieval_result
        )
    
    def tune_hyperparameters(self, graph_vector_weight: float = None, 
                           expansion_depth: int = None, 
                           edge_type_boosts: Dict[str, float] = None):
        """Tune hyperparameters for graph-aware retrieval"""
        
        if graph_vector_weight is not None:
            self.graph_vector_weight = graph_vector_weight
        
        if expansion_depth is not None:
            self.expansion_depth = expansion_depth
        
        if edge_type_boosts is not None:
            self.edge_type_boosts.update(edge_type_boosts)
        
        # Update retrieval system parameters
        if hasattr(self.semantic_graph_agent.retrieval_system, 'vector_weight'):
            self.semantic_graph_agent.retrieval_system.vector_weight = self.graph_vector_weight
            self.semantic_graph_agent.retrieval_system.graph_weight = 1.0 - self.graph_vector_weight
        
        if hasattr(self.semantic_graph_agent.retrieval_system, 'expansion_depth'):
            self.semantic_graph_agent.retrieval_system.expansion_depth = self.expansion_depth

class EvaluationFramework:
    """Main evaluation framework for comparing RAG systems"""
    
    def __init__(self):
        self.pure_rag = None
        self.graph_rag = None
        self.dataset = None
        self.results = {}
        
    def setup_systems(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Setup both RAG systems"""
        
        print("Setting up Pure RAG system...")
        self.pure_rag = PureRAGSystem(embedding_model)
        
        print("Setting up Graph-Aware RAG system...")
        self.graph_rag = GraphAwareRAGSystem(embedding_model)
    
    def load_or_generate_dataset(self, dataset_path: str = None, corpus_size: int = 10000, num_queries: int = 500):
        """Load existing dataset or generate new one"""
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            with open(os.path.join(dataset_path, "passages.json"), "r") as f:
                passages = json.load(f)
            with open(os.path.join(dataset_path, "queries.json"), "r") as f:
                queries_data = json.load(f)
                queries = [EvaluationQuery(**q) for q in queries_data]
        else:
            print("Generating new dataset...")
            generator = DatasetGenerator()
            passages = generator.generate_synthetic_corpus(corpus_size)
            queries = generator.generate_evaluation_queries(num_queries)
            
            # Save dataset
            if dataset_path:
                generator.save_dataset(dataset_path)
        
        self.dataset = {"passages": passages, "queries": queries}
        print(f"Dataset loaded: {len(passages)} passages, {len(queries)} queries")
        
        return passages, queries
    
    def build_indices(self, passages: List[Dict[str, Any]]):
        """Build indices for both systems"""
        
        print("Building Pure RAG index...")
        self.pure_rag.build_index(passages)
        
        print("Building Graph-Aware RAG index...")
        self.graph_rag.build_graph_index(passages)
    
    def evaluate_system(self, system, queries: List[EvaluationQuery], system_name: str) -> Dict[str, Any]:
        """Evaluate a single system on all queries"""
        
        print(f"Evaluating {system_name}...")
        
        results = []
        retrieval_times = []
        generation_times = []
        
        for i, query in enumerate(queries):
            if i % 50 == 0:
                print(f"Processing query {i}/{len(queries)}")
            
            try:
                # Retrieve passages
                retrieval_result = system.retrieve(query.question, k=10)
                retrieval_result.query_id = query.id
                retrieval_times.append(retrieval_result.retrieval_time)
                
                # Generate answer
                answer_result = system.generate_answer(query.question, retrieval_result)
                generation_times.append(answer_result.generation_time)
                
                results.append(answer_result)
                
            except Exception as e:
                print(f"Error processing query {query.id}: {e}")
                # Create dummy result for failed queries
                dummy_retrieval = RetrievalResult(
                    query_id=query.id,
                    retrieved_passages=[],
                    retrieval_time=0.0,
                    method=system_name,
                    parameters={}
                )
                dummy_answer = AnswerResult(
                    query_id=query.id,
                    generated_answer="Error processing query",
                    generation_time=0.0,
                    retrieval_result=dummy_retrieval
                )
                results.append(dummy_answer)
        
        return {
            "results": results,
            "avg_retrieval_time": np.mean(retrieval_times) if retrieval_times else 0.0,
            "avg_generation_time": np.mean(generation_times) if generation_times else 0.0
        }
    
    def calculate_metrics(self, system_results: Dict[str, Any], queries: List[EvaluationQuery]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        
        results = system_results["results"]
        
        # Create query lookup
        query_lookup = {q.id: q for q in queries}
        
        # Initialize metrics
        recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_scores = []
        exact_matches = 0
        f1_scores = []
        hallucination_count = 0
        
        for result in results:
            query = query_lookup.get(result.query_id)
            if not query:
                continue
            
            retrieved_passage_ids = [p.get("id", "") for p in result.retrieval_result.retrieved_passages]
            relevant_passage_ids = set(query.relevant_passages)
            
            # Calculate Recall@k
            for k in recall_at_k.keys():
                top_k_ids = set(retrieved_passage_ids[:k])
                if top_k_ids & relevant_passage_ids:
                    recall_at_k[k] += 1
            
            # Calculate MRR
            reciprocal_rank = 0
            for i, passage_id in enumerate(retrieved_passage_ids):
                if passage_id in relevant_passage_ids:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            mrr_scores.append(reciprocal_rank)
            
            # Calculate Exact Match
            if self._normalize_answer(result.generated_answer) == self._normalize_answer(query.expected_answer):
                exact_matches += 1
            
            # Calculate F1 Score
            f1 = self._calculate_f1(result.generated_answer, query.expected_answer)
            f1_scores.append(f1)
            
            # Check for hallucination (simplified)
            if self._check_hallucination(result.generated_answer, result.retrieval_result.retrieved_passages):
                hallucination_count += 1
        
        # Normalize metrics
        total_queries = len(results)
        for k in recall_at_k:
            recall_at_k[k] = recall_at_k[k] / total_queries if total_queries > 0 else 0.0
        
        return EvaluationMetrics(
            recall_at_k=recall_at_k,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            exact_match=exact_matches / total_queries if total_queries > 0 else 0.0,
            f1_score=np.mean(f1_scores) if f1_scores else 0.0,
            hallucination_rate=hallucination_count / total_queries if total_queries > 0 else 0.0,
            avg_retrieval_time=system_results["avg_retrieval_time"],
            avg_generation_time=system_results["avg_generation_time"],
            total_queries=total_queries
        )
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for exact match comparison"""
        return re.sub(r'[^\w\s]', '', answer.lower().strip())
    
    def _calculate_f1(self, generated: str, reference: str) -> float:
        """Calculate F1 score between generated and reference answers"""
        
        gen_tokens = set(self._normalize_answer(generated).split())
        ref_tokens = set(self._normalize_answer(reference).split())
        
        if not gen_tokens and not ref_tokens:
            return 1.0
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        intersection = gen_tokens & ref_tokens
        precision = len(intersection) / len(gen_tokens)
        recall = len(intersection) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _check_hallucination(self, answer: str, retrieved_passages: List[Dict[str, Any]]) -> bool:
        """Simple hallucination check - whether answer contains unsupported statements"""
        
        # Extract key terms from answer
        answer_terms = set(self._normalize_answer(answer).split())
        
        # Extract terms from retrieved passages
        passage_terms = set()
        for passage in retrieved_passages:
            content = passage.get("content", "")
            passage_terms.update(self._normalize_answer(content).split())
        
        # Check if answer has terms not in passages (simplified hallucination detection)
        unsupported_terms = answer_terms - passage_terms
        
        # Consider it hallucination if more than 30% of terms are unsupported
        if len(answer_terms) > 0:
            hallucination_ratio = len(unsupported_terms) / len(answer_terms)
            return hallucination_ratio > 0.3
        
        return False
    
    def run_hyperparameter_tuning(self, queries: List[EvaluationQuery]) -> Dict[str, Any]:
        """Run hyperparameter tuning for graph-aware system"""
        
        print("Running hyperparameter tuning...")
        
        # Parameter ranges
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Graph vs Vector weight
        depth_values = [1, 2, 3]  # Expansion depth
        
        best_config = None
        best_score = 0.0
        tuning_results = []
        
        # Sample smaller set of queries for tuning
        tuning_queries = queries[:100]  # Use first 100 queries for tuning
        
        for alpha in alpha_values:
            for depth in depth_values:
                print(f"Testing α={alpha}, depth={depth}")
                
                # Configure hyperparameters
                self.graph_rag.tune_hyperparameters(
                    graph_vector_weight=alpha,
                    expansion_depth=depth
                )
                
                # Evaluate on tuning set
                system_results = self.evaluate_system(self.graph_rag, tuning_queries, f"graph_tuning_a{alpha}_d{depth}")
                metrics = self.calculate_metrics(system_results, tuning_queries)
                
                # Combined score (weighted average of key metrics)
                combined_score = (
                    0.3 * metrics.recall_at_k[5] +
                    0.3 * metrics.mrr +
                    0.2 * metrics.f1_score +
                    0.2 * (1.0 - metrics.hallucination_rate)
                )
                
                tuning_results.append({
                    "alpha": alpha,
                    "depth": depth,
                    "combined_score": combined_score,
                    "recall_at_5": metrics.recall_at_k[5],
                    "mrr": metrics.mrr,
                    "f1_score": metrics.f1_score,
                    "hallucination_rate": metrics.hallucination_rate
                })
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_config = {"alpha": alpha, "depth": depth}
        
        # Apply best configuration
        if best_config:
            print(f"Best configuration: α={best_config['alpha']}, depth={best_config['depth']}")
            self.graph_rag.tune_hyperparameters(
                graph_vector_weight=best_config["alpha"],
                expansion_depth=best_config["depth"]
            )
        
        return {
            "best_config": best_config,
            "best_score": best_score,
            "all_results": tuning_results
        }
    
    def run_full_evaluation(self, dataset_path: str = None, corpus_size: int = 10000, 
                          num_queries: int = 500, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Run complete evaluation comparing both systems"""
        
        print("🚀 Starting RAG vs Graph-Aware Retrieval Evaluation")
        print("=" * 60)
        
        # Setup
        self.setup_systems()
        passages, queries = self.load_or_generate_dataset(dataset_path, corpus_size, num_queries)
        self.build_indices(passages)
        
        # Hyperparameter tuning for graph system
        tuning_results = None
        if tune_hyperparameters:
            tuning_results = self.run_hyperparameter_tuning(queries)
        
        # Evaluate both systems
        print("\n📊 Running Full Evaluation...")
        
        pure_rag_results = self.evaluate_system(self.pure_rag, queries, "Pure RAG")
        graph_rag_results = self.evaluate_system(self.graph_rag, queries, "Graph-Aware RAG")
        
        # Calculate metrics
        pure_rag_metrics = self.calculate_metrics(pure_rag_results, queries)
        graph_rag_metrics = self.calculate_metrics(graph_rag_results, queries)
        
        # Compile results
        evaluation_results = {
            "dataset_info": {
                "num_passages": len(passages),
                "num_queries": len(queries),
                "corpus_size": corpus_size
            },
            "pure_rag": {
                "metrics": asdict(pure_rag_metrics),
                "system_info": {
                    "method": "FAISS + Vector Similarity",
                    "embedding_model": self.pure_rag.embedding_model_name if hasattr(self.pure_rag, 'embedding_model_name') else "all-MiniLM-L6-v2"
                }
            },
            "graph_aware_rag": {
                "metrics": asdict(graph_rag_metrics),
                "system_info": {
                    "method": "Semantic Graph + Vector Hybrid",
                    "graph_vector_weight": self.graph_rag.graph_vector_weight,
                    "expansion_depth": self.graph_rag.expansion_depth,
                    "edge_type_boosts": self.graph_rag.edge_type_boosts
                }
            },
            "hyperparameter_tuning": tuning_results,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Calculate improvements
        improvements = self._calculate_improvements(pure_rag_metrics, graph_rag_metrics)
        evaluation_results["improvements"] = improvements
        
        return evaluation_results
    
    def _calculate_improvements(self, pure_metrics: EvaluationMetrics, 
                              graph_metrics: EvaluationMetrics) -> Dict[str, float]:
        """Calculate percentage improvements of graph system over pure RAG"""
        
        improvements = {}
        
        # Recall improvements
        for k in [1, 3, 5, 10]:
            if pure_metrics.recall_at_k[k] > 0:
                improvement = ((graph_metrics.recall_at_k[k] - pure_metrics.recall_at_k[k]) / 
                             pure_metrics.recall_at_k[k]) * 100
                improvements[f"recall_at_{k}"] = improvement
        
        # Other metric improvements
        if pure_metrics.mrr > 0:
            improvements["mrr"] = ((graph_metrics.mrr - pure_metrics.mrr) / pure_metrics.mrr) * 100
        
        if pure_metrics.f1_score > 0:
            improvements["f1_score"] = ((graph_metrics.f1_score - pure_metrics.f1_score) / pure_metrics.f1_score) * 100
        
        improvements["exact_match"] = ((graph_metrics.exact_match - pure_metrics.exact_match) / 
                                     max(pure_metrics.exact_match, 0.01)) * 100
        
        # Hallucination rate improvement (lower is better)
        if pure_metrics.hallucination_rate > 0:
            improvements["hallucination_rate"] = ((pure_metrics.hallucination_rate - graph_metrics.hallucination_rate) / 
                                                pure_metrics.hallucination_rate) * 100
        
        return improvements
    
    def generate_report(self, results: Dict[str, Any], output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        
        # Save detailed results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n📈 EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        pure_metrics = results["pure_rag"]["metrics"]
        graph_metrics = results["graph_aware_rag"]["metrics"]
        improvements = results["improvements"]
        
        print(f"\n🔍 RETRIEVAL PERFORMANCE:")
        print(f"                    Pure RAG    Graph-Aware    Improvement")
        print(f"Recall@1:           {pure_metrics['recall_at_k']['1']:.3f}       {graph_metrics['recall_at_k']['1']:.3f}        {improvements.get('recall_at_1', 0):.1f}%")
        print(f"Recall@5:           {pure_metrics['recall_at_k']['5']:.3f}       {graph_metrics['recall_at_k']['5']:.3f}        {improvements.get('recall_at_5', 0):.1f}%")
        print(f"Recall@10:          {pure_metrics['recall_at_k']['10']:.3f}       {graph_metrics['recall_at_k']['10']:.3f}        {improvements.get('recall_at_10', 0):.1f}%")
        print(f"MRR:                {pure_metrics['mrr']:.3f}       {graph_metrics['mrr']:.3f}        {improvements.get('mrr', 0):.1f}%")
        
        print(f"\n📝 ANSWER QUALITY:")
        print(f"Exact Match:        {pure_metrics['exact_match']:.3f}       {graph_metrics['exact_match']:.3f}        {improvements.get('exact_match', 0):.1f}%")
        print(f"F1 Score:           {pure_metrics['f1_score']:.3f}       {graph_metrics['f1_score']:.3f}        {improvements.get('f1_score', 0):.1f}%")
        print(f"Hallucination Rate: {pure_metrics['hallucination_rate']:.3f}       {graph_metrics['hallucination_rate']:.3f}        {improvements.get('hallucination_rate', 0):.1f}%")
        
        print(f"\n⏱️ PERFORMANCE:")
        print(f"Avg Retrieval Time: {pure_metrics['avg_retrieval_time']:.3f}s      {graph_metrics['avg_retrieval_time']:.3f}s")
        print(f"Avg Generation Time:{pure_metrics['avg_generation_time']:.3f}s      {graph_metrics['avg_generation_time']:.3f}s")
        
        # Hyperparameter tuning results
        if results.get("hyperparameter_tuning"):
            tuning = results["hyperparameter_tuning"]
            print(f"\n🎛️ HYPERPARAMETER TUNING:")
            print(f"Best Configuration: α={tuning['best_config']['alpha']}, depth={tuning['best_config']['depth']}")
            print(f"Best Combined Score: {tuning['best_score']:.3f}")
        
        print(f"\n💾 Detailed results saved to: {output_path}")
        
        return results

def main():
    """Main evaluation execution"""
    
    # Check dependencies
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Please install: pip install faiss-cpu sentence-transformers")
        return
    
    # Create evaluation framework
    evaluator = EvaluationFramework()
    
    # Run evaluation
    try:
        results = evaluator.run_full_evaluation(
            dataset_path="evaluation/dataset",
            corpus_size=5000,  # Smaller for demo
            num_queries=200,   # Smaller for demo
            tune_hyperparameters=True
        )
        
        # Generate report
        evaluator.generate_report(results, "evaluation/rag_vs_graph_evaluation_results.json")
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()