# -*- coding: utf-8 -*-
"""
Advanced Memory Manager for AI Research Agent
Integrates hierarchical memory with LangMem and provides research-specific tools
"""

from typing import Dict, List, Any, Optional, Tuple
from langchain.tools import Tool
from memory.hierarchical_memory import HierarchicalMemory, MemoryType, MemoryNode
from memory.langmem_tools import get_memory_tools
import json
import re
from datetime import datetime
import torch
import logging

# Diffusion integration
try:
    from diffusion.research_agent_integration import create_diffusion_enhanced_agent, DiffusionIntegrationConfig
    from diffusion.synthetic_data_generator import SyntheticDataGenerator
    from diffusion.denoising_layer import DenoisingLayer
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedMemoryManager:
    """Advanced memory manager with hierarchical memory, citation tracking, and diffusion enhancement"""
    
    def __init__(self, enable_diffusion: bool = True):
        self.hierarchical_memory = HierarchicalMemory()
        self.langmem_tools = get_memory_tools()
        self.current_episode_id = None
        
        # Diffusion enhancement
        self.enable_diffusion = enable_diffusion and DIFFUSION_AVAILABLE
        self.diffusion_agent = None
        self.synthetic_contexts_cache = {}
        
        if self.enable_diffusion:
            try:
                self.diffusion_agent = create_diffusion_enhanced_agent(enable_all=True)
                logger.info("Diffusion-enhanced memory manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize diffusion agent: {e}")
                self.enable_diffusion = False
        
    def start_research_session(self, question: str) -> str:
        """Start a new research session with episodic memory tracking"""
        self.current_episode_id = self.hierarchical_memory.create_research_episode(question)
        return self.current_episode_id
    
    def save_research_finding(self, content: str, importance: float = 0.5,
                            citations: List[str] = None, concepts: List[str] = None,
                            step_info: Dict[str, Any] = None) -> str:
        """Save a research finding with enhanced metadata"""
        
        # Extract concepts if not provided
        if concepts is None:
            concepts = self._extract_concepts(content)
        
        # Extract citations if not provided
        if citations is None:
            citations = self._extract_citations(content)
        
        # Prepare metadata
        metadata = {
            "research_session": self.current_episode_id,
            "timestamp": datetime.now().isoformat(),
            "step_info": step_info or {},
            "content_type": "research_finding"
        }
        
        # Save to hierarchical memory
        memory_id = self.hierarchical_memory.add_memory(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            importance=importance,
            metadata=metadata,
            citations=citations,
            concepts=concepts
        )
        
        # Also save to LangMem for semantic search
        langmem_content = f"Research Finding: {content}"
        if citations:
            langmem_content += f"\nCitations: {', '.join(citations)}"
        if concepts:
            langmem_content += f"\nConcepts: {', '.join(concepts)}"
        
        self.langmem_tools[0].invoke(langmem_content)
        
        # Update current episode
        if self.current_episode_id:
            finding_data = {
                "memory_id": memory_id,
                "content": content,
                "concepts": concepts,
                "citations": citations,
                "timestamp": datetime.now().isoformat()
            }
            self.hierarchical_memory.update_research_episode(
                self.current_episode_id, finding=finding_data, citations=citations
            )
        
        return memory_id
    
    def search_research_memory(self, query: str, include_citations: bool = True,
                             memory_types: List[MemoryType] = None) -> Dict[str, Any]:
        """Enhanced memory search with citation tracking"""
        
        # Search hierarchical memory
        hierarchical_results = self.hierarchical_memory.search_memory(
            query, memory_types=memory_types
        )
        
        # Search LangMem for semantic similarity
        langmem_result = self.langmem_tools[1].invoke(query)
        
        # Combine and format results
        formatted_results = {
            "query": query,
            "hierarchical_matches": [],
            "semantic_matches": langmem_result,
            "related_concepts": [],
            "citations": [],
            "knowledge_graph_insights": {}
        }
        
        # Process hierarchical results
        for memory, relevance in hierarchical_results:
            result_item = {
                "content": memory.content,
                "relevance_score": relevance,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance_score,
                "concepts": memory.related_concepts,
                "citations": memory.citations,
                "timestamp": memory.timestamp.isoformat(),
                "access_count": memory.access_count
            }
            formatted_results["hierarchical_matches"].append(result_item)
            
            # Collect related concepts and citations
            formatted_results["related_concepts"].extend(memory.related_concepts)
            if include_citations:
                formatted_results["citations"].extend(memory.citations)
        
        # Remove duplicates
        formatted_results["related_concepts"] = list(set(formatted_results["related_concepts"]))
        formatted_results["citations"] = list(set(formatted_results["citations"]))
        
        # Get knowledge graph insights for top concepts
        top_concepts = formatted_results["related_concepts"][:3]
        for concept in top_concepts:
            insights = self.hierarchical_memory.get_knowledge_graph_insights(concept)
            if "error" not in insights:
                formatted_results["knowledge_graph_insights"][concept] = insights
        
        return formatted_results
    
    def save_hypothesis(self, hypothesis: str, supporting_evidence: List[str] = None,
                       confidence: float = 0.5) -> str:
        """Save a research hypothesis with supporting evidence"""
        
        content = f"Hypothesis: {hypothesis}"
        if supporting_evidence:
            content += f"\nSupporting Evidence: {'; '.join(supporting_evidence)}"
        
        concepts = self._extract_concepts(hypothesis)
        concepts.append("hypothesis")  # Tag as hypothesis
        
        metadata = {
            "content_type": "hypothesis",
            "confidence": confidence,
            "supporting_evidence": supporting_evidence or [],
            "research_session": self.current_episode_id
        }
        
        return self.hierarchical_memory.add_memory(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            importance=confidence,
            metadata=metadata,
            concepts=concepts
        )
    
    def consolidate_session_memories(self) -> Dict[str, Any]:
        """Consolidate memories from current session"""
        
        promoted_count = self.hierarchical_memory.consolidate_memories()
        stats = self.hierarchical_memory.get_memory_statistics()
        
        return {
            "memories_promoted": promoted_count,
            "memory_statistics": stats,
            "session_id": self.current_episode_id
        }
    
    def end_research_session(self, final_answer: str = "") -> Dict[str, Any]:
        """End current research session and consolidate findings"""
        
        if self.current_episode_id:
            # Update episode with final answer
            self.hierarchical_memory.update_research_episode(
                self.current_episode_id, final_answer=final_answer
            )
            
            # Get episode summary
            episode = self.hierarchical_memory.episodic_memory[self.current_episode_id]
            
            # Consolidate memories
            consolidation_results = self.consolidate_session_memories()
            
            session_summary = {
                "episode_id": self.current_episode_id,
                "question": episode.question,
                "duration": (episode.end_time - episode.start_time).total_seconds() if episode.end_time else 0,
                "findings_count": len(episode.findings),
                "citations_used": len(episode.citations_used),
                "final_answer": final_answer,
                "consolidation_results": consolidation_results
            }
            
            self.current_episode_id = None
            return session_summary
        
        return {"error": "No active research session"}
    
    def get_citation_network(self, citation: str) -> Dict[str, Any]:
        """Get network of memories connected to a specific citation"""
        
        if citation not in self.hierarchical_memory.citation_index:
            return {"error": f"Citation '{citation}' not found"}
        
        memory_ids = self.hierarchical_memory.citation_index[citation]
        connected_memories = []
        
        for memory_id in memory_ids:
            # Check both short-term and long-term memory
            memory = None
            for stm in self.hierarchical_memory.short_term_memory:
                if stm.id == memory_id:
                    memory = stm
                    break
            
            if not memory and memory_id in self.hierarchical_memory.long_term_memory:
                memory = self.hierarchical_memory.long_term_memory[memory_id]
            
            if memory:
                connected_memories.append({
                    "content": memory.content,
                    "concepts": memory.related_concepts,
                    "importance": memory.importance_score,
                    "timestamp": memory.timestamp.isoformat()
                })
        
        return {
            "citation": citation,
            "connected_memories": connected_memories,
            "connection_count": len(connected_memories)
        }
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using simple NLP"""
        
        # Simple concept extraction - can be enhanced with NLP libraries
        concepts = []
        
        # Look for capitalized words (potential proper nouns/concepts)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.extend(capitalized_words)
        
        # Look for technical terms (words with specific patterns)
        technical_terms = re.findall(r'\b\w*(?:tion|ism|ology|graphy|metry)\b', text, re.IGNORECASE)
        concepts.extend(technical_terms)
        
        # Look for quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', text)
        concepts.extend(quoted_terms)
        
        # Remove duplicates and filter
        concepts = list(set([c for c in concepts if len(c) > 2]))
        
        return concepts[:10]  # Limit to top 10 concepts
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        
        citations = []
        
        # Look for URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        citations.extend(urls)
        
        # Look for DOIs
        dois = re.findall(r'10\.\d{4,}\/[^\s]+', text)
        citations.extend([f"doi:{doi}" for doi in dois])
        
        # Look for author-year citations
        author_year = re.findall(r'\b[A-Z][a-z]+\s+et\s+al\.\s+\(\d{4}\)', text)
        citations.extend(author_year)
        
        # Look for simple author-year format
        simple_citations = re.findall(r'\b[A-Z][a-z]+\s+\(\d{4}\)', text)
        citations.extend(simple_citations)
        
        return list(set(citations))

def get_advanced_memory_tools():
    """Get advanced memory tools for research agent"""
    
    memory_manager = AdvancedMemoryManager()
    
    def save_finding_tool(content: str) -> str:
        """Save research finding with automatic concept and citation extraction"""
        try:
            memory_id = memory_manager.save_research_finding(content)
            return f"Research finding saved successfully. Memory ID: {memory_id}"
        except Exception as e:
            return f"Error saving research finding: {str(e)}"
    
    def search_memory_tool(query: str) -> str:
        """Search memory with enhanced results including citations and concepts"""
        try:
            results = memory_manager.search_research_memory(query)
            
            # Format results for display
            formatted_output = f"Memory search results for: {query}\n"
            formatted_output += "=" * 50 + "\n"
            
            if results["hierarchical_matches"]:
                formatted_output += "Hierarchical Memory Matches:\n"
                for i, match in enumerate(results["hierarchical_matches"][:3], 1):
                    formatted_output += f"{i}. {match['content'][:200]}...\n"
                    formatted_output += f"   Relevance: {match['relevance_score']:.2f}, Type: {match['memory_type']}\n"
                    if match['concepts']:
                        formatted_output += f"   Concepts: {', '.join(match['concepts'][:3])}\n"
                    formatted_output += "\n"
            
            if results["semantic_matches"]:
                formatted_output += f"Semantic Matches:\n{results['semantic_matches']}\n\n"
            
            if results["related_concepts"]:
                formatted_output += f"Related Concepts: {', '.join(results['related_concepts'][:5])}\n"
            
            if results["citations"]:
                formatted_output += f"Related Citations: {', '.join(results['citations'][:3])}\n"
            
            return formatted_output
            
        except Exception as e:
            return f"Error searching memory: {str(e)}"
    
    def start_session_tool(question: str) -> str:
        """Start a new research session"""
        try:
            session_id = memory_manager.start_research_session(question)
            return f"Research session started. Session ID: {session_id}"
        except Exception as e:
            return f"Error starting research session: {str(e)}"
    
    def end_session_tool(final_answer: str = "") -> str:
        """End current research session"""
        try:
            summary = memory_manager.end_research_session(final_answer)
            if "error" in summary:
                return summary["error"]
            
            return f"Research session ended. Findings: {summary['findings_count']}, Citations: {summary['citations_used']}"
        except Exception as e:
            return f"Error ending research session: {str(e)}"
    
    return [
        Tool(
            name="save_research_finding",
            description="Save important research findings with automatic concept and citation extraction",
            func=save_finding_tool
        ),
        Tool(
            name="search_advanced_memory",
            description="Search memory with enhanced results including hierarchical matching, concepts, and citations",
            func=search_memory_tool
        ),
        Tool(
            name="start_research_session",
            description="Start a new research session with episodic memory tracking",
            func=start_session_tool
        ),
        Tool(
            name="end_research_session",
            description="End current research session and consolidate findings",
            func=end_session_tool
        )
    ]    
   
 # Diffusion-Enhanced Memory Methods
    
    def initialize_diffusion_training(self, training_contexts: List[str]):
        """Initialize diffusion models with memory contexts"""
        if not self.enable_diffusion or not self.diffusion_agent:
            logger.warning("Diffusion not available for memory enhancement")
            return False
            
        try:
            self.diffusion_agent.initialize_with_training_data(training_contexts)
            logger.info(f"Diffusion models initialized with {len(training_contexts)} memory contexts")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize diffusion training: {e}")
            return False
    
    def augment_memory_contexts(self, contexts: List[str], 
                              augmentation_ratio: float = 2.0) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Augment memory contexts using diffusion-based paraphrase generation"""
        if not self.enable_diffusion or not self.diffusion_agent:
            logger.warning("Diffusion not available - returning original contexts")
            return contexts, [{'synthetic': False} for _ in contexts]
            
        try:
            # Check cache first
            cache_key = f"{len(contexts)}_{augmentation_ratio}"
            if cache_key in self.synthetic_contexts_cache:
                logger.debug("Using cached synthetic contexts")
                return self.synthetic_contexts_cache[cache_key]
            
            # Generate synthetic contexts
            augmented_contexts, metadata = self.diffusion_agent.augment_memory_contexts(contexts)
            
            # Cache results
            self.synthetic_contexts_cache[cache_key] = (augmented_contexts, metadata)
            
            logger.info(f"Augmented {len(contexts)} contexts to {len(augmented_contexts)} using diffusion")
            return augmented_contexts, metadata
            
        except Exception as e:
            logger.error(f"Failed to augment memory contexts: {e}")
            return contexts, [{'synthetic': False, 'error': str(e)} for _ in contexts]
    
    def enhanced_memory_retrieval(self, query: str, top_k: int = 5, 
                                use_denoising: bool = True) -> Dict[str, Any]:
        """Enhanced memory retrieval with diffusion-based denoising"""
        if not self.enable_diffusion or not self.diffusion_agent:
            # Fallback to regular search
            return self.search_research_memory(query)
            
        try:
            # Get all memory contents for embedding
            all_memories = []
            memory_metadata = []
            
            # Collect from hierarchical memory
            for memory in self.hierarchical_memory.short_term_memory:
                all_memories.append(memory.content)
                memory_metadata.append({
                    'id': memory.id,
                    'type': 'short_term',
                    'importance': memory.importance_score,
                    'concepts': memory.related_concepts,
                    'citations': memory.citations
                })
            
            for memory_id, memory in self.hierarchical_memory.long_term_memory.items():
                all_memories.append(memory.content)
                memory_metadata.append({
                    'id': memory_id,
                    'type': 'long_term',
                    'importance': memory.importance_score,
                    'concepts': memory.related_concepts,
                    'citations': memory.citations
                })
            
            if not all_memories:
                return {"error": "No memories available for retrieval"}
            
            # Encode memories and query
            memory_embeddings = self.diffusion_agent.diffusion_core.encode_text(all_memories)
            query_embedding = self.diffusion_agent.diffusion_core.encode_text([query])
            
            # Use diffusion-enhanced retrieval
            if use_denoising:
                retrieval_results = self.diffusion_agent.robust_retrieval_query(
                    query_embedding, memory_embeddings, top_k
                )
            else:
                # Simple cosine similarity
                similarities = torch.cosine_similarity(
                    query_embedding, memory_embeddings, dim=1
                )
                top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(similarities)))
                retrieval_results = {
                    'retrieved_indices': top_k_indices.tolist(),
                    'similarities': top_k_values.tolist(),
                    'denoised': False
                }
            
            # Format results
            enhanced_results = {
                'query': query,
                'diffusion_enhanced': True,
                'denoised': retrieval_results.get('denoised', False),
                'coherence_score': retrieval_results.get('coherence_score', 0.0),
                'retrieved_memories': []
            }
            
            for i, idx in enumerate(retrieval_results['retrieved_indices']):
                memory_data = memory_metadata[idx]
                memory_data['similarity_score'] = retrieval_results['similarities'][i]
                memory_data['content'] = all_memories[idx]
                enhanced_results['retrieved_memories'].append(memory_data)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Enhanced memory retrieval failed: {e}")
            # Fallback to regular search
            return self.search_research_memory(query)
    
    def generate_memory_insights(self, concept: str) -> Dict[str, Any]:
        """Generate creative insights about a concept using diffusion-based idea exploration"""
        if not self.enable_diffusion or not self.diffusion_agent:
            return {"error": "Diffusion not available for insight generation"}
            
        try:
            # Search for related memories first
            related_memories = self.search_research_memory(concept)
            
            # Create exploration context
            exploration_context = {
                'related_concepts': related_memories.get('related_concepts', []),
                'citations': related_memories.get('citations', []),
                'memory_count': len(related_memories.get('hierarchical_matches', []))
            }
            
            # Generate creative insights
            creative_insights = self.diffusion_agent.explore_creative_ideas(
                f"Research insights about {concept}", exploration_context
            )
            
            return {
                'concept': concept,
                'related_memories': related_memories,
                'creative_insights': creative_insights,
                'generation_method': 'diffusion_enhanced'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate memory insights: {e}")
            return {"error": str(e)}
    
    def create_synthetic_research_examples(self, topic: str, num_examples: int = 5) -> List[Dict[str, Any]]:
        """Create synthetic research examples for a topic using diffusion"""
        if not self.enable_diffusion or not self.diffusion_agent:
            return []
            
        try:
            # Get existing memories about the topic
            existing_memories = self.search_research_memory(topic)
            
            # Prepare training examples
            training_examples = []
            for memory in existing_memories.get('hierarchical_matches', [])[:10]:
                training_examples.append({
                    'text': memory['content'],
                    'concepts': memory['concepts'],
                    'importance': memory['importance']
                })
            
            if not training_examples:
                # Create basic examples if no memories exist
                training_examples = [{
                    'text': f"Research about {topic} shows interesting patterns and insights.",
                    'concepts': [topic],
                    'importance': 0.5
                }]
            
            # Generate synthetic examples
            synthetic_examples = self.diffusion_agent.balance_training_dataset(training_examples)
            
            # Filter for synthetic ones
            synthetic_only = [ex for ex in synthetic_examples if ex.get('synthetic', False)]
            
            return synthetic_only[:num_examples]
            
        except Exception as e:
            logger.error(f"Failed to create synthetic research examples: {e}")
            return []
    
    def get_diffusion_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about diffusion-enhanced memory operations"""
        stats = {
            'diffusion_enabled': self.enable_diffusion,
            'diffusion_available': DIFFUSION_AVAILABLE,
            'synthetic_contexts_cached': len(self.synthetic_contexts_cache),
            'diffusion_agent_initialized': self.diffusion_agent is not None
        }
        
        if self.diffusion_agent:
            stats.update(self.diffusion_agent.get_comprehensive_stats())
            
        return stats
