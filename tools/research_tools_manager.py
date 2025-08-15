# -*- coding: utf-8 -*-
"""
Research Tools Manager for AI Research Agent
Integrates and manages all Phase 3 research tools
"""

from typing import List, Dict, Any
from langchain.tools import Tool
from tools.web_research import get_web_research_tools
from tools.document_processor import get_document_processing_tools
from tools.data_visualization import get_visualization_tools
import logging
import pandas as pd

# Diffusion integration
try:
    from diffusion.research_agent_integration import create_diffusion_enhanced_agent
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False

logger = logging.getLogger(__name__)

class ResearchToolsManager:
    """Manages all research tools and provides intelligent tool selection with diffusion enhancement"""
    
    def __init__(self, enable_diffusion: bool = True):
        self.web_tools = get_web_research_tools()
        self.document_tools = get_document_processing_tools()
        self.visualization_tools = get_visualization_tools()
        
        # Initialize diffusion tools
        self.enable_diffusion = enable_diffusion and DIFFUSION_AVAILABLE
        self.diffusion_tools = []
        self.diffusion_agent = None
        
        if self.enable_diffusion:
            try:
                self.diffusion_agent = create_diffusion_enhanced_agent(enable_all=True)
                self.diffusion_tools = self._create_diffusion_tools()
                logger.info("Diffusion tools initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize diffusion tools: {e}")
                self.enable_diffusion = False
        
        # Combine all tools
        self.all_tools = (
            self.web_tools + 
            self.document_tools + 
            self.visualization_tools +
            self.diffusion_tools
        )
        
        # Create tool categories
        self.tool_categories = {
            'web_research': [tool.name for tool in self.web_tools],
            'document_processing': [tool.name for tool in self.document_tools],
            'data_visualization': [tool.name for tool in self.visualization_tools],
            'diffusion_enhanced': [tool.name for tool in self.diffusion_tools]
        }
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available research tools"""
        return self.all_tools
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools by category"""
        if category not in self.tool_categories:
            return []
        
        tool_names = self.tool_categories[category]
        return [tool for tool in self.all_tools if tool.name in tool_names]
    
    def suggest_tools_for_query(self, query: str) -> Dict[str, List[str]]:
        """Suggest appropriate tools based on research query"""
        query_lower = query.lower()
        suggestions = {
            'recommended': [],
            'optional': [],
            'visualization': []
        }
        
        # Web research suggestions
        if any(keyword in query_lower for keyword in ['current', 'recent', 'latest', 'news', 'today']):
            suggestions['recommended'].extend(['web_search', 'news_search'])
        
        if any(keyword in query_lower for keyword in ['academic', 'research', 'paper', 'study']):
            suggestions['recommended'].append('arxiv_search')
        
        if any(keyword in query_lower for keyword in ['definition', 'what is', 'explain', 'background']):
            suggestions['recommended'].append('wikipedia_search')
        
        # Document processing suggestions
        if any(keyword in query_lower for keyword in ['pdf', 'document', 'paper', 'article']):
            suggestions['optional'].extend(['process_pdf_url', 'analyze_document_structure'])
        
        # Visualization suggestions
        if any(keyword in query_lower for keyword in ['trend', 'timeline', 'over time', 'history']):
            suggestions['visualization'].append('create_timeline_visualization')
        
        if any(keyword in query_lower for keyword in ['relationship', 'connection', 'network', 'concept']):
            suggestions['visualization'].append('create_concept_network')
        
        if any(keyword in query_lower for keyword in ['analysis', 'frequency', 'common', 'popular']):
            suggestions['visualization'].append('create_word_frequency_chart')
        
        # Default suggestions if none match
        if not any(suggestions.values()):
            suggestions['recommended'] = ['web_search', 'wikipedia_search']
            suggestions['optional'] = ['arxiv_search']
        
        # Add diffusion tool suggestions
        if self.enable_diffusion:
            if any(keyword in query_lower for keyword in ['creative', 'brainstorm', 'ideas', 'innovative']):
                suggestions['recommended'].append('explore_creative_ideas')
            
            if any(keyword in query_lower for keyword in ['visual', 'webpage', 'screenshot', 'image']):
                suggestions['recommended'].append('analyze_webpage_visually')
            
            if any(keyword in query_lower for keyword in ['plan', 'strategy', 'approach', 'methodology']):
                suggestions['recommended'].append('generate_diverse_research_plan')
        
        return suggestions
    
    def _create_diffusion_tools(self) -> List[Tool]:
        """Create diffusion-enhanced research tools"""
        if not self.diffusion_agent:
            return []
        
        def explore_creative_ideas_tool(prompt: str) -> str:
            """Explore creative research ideas using diffusion-based brainstorming"""
            try:
                result = self.diffusion_agent.explore_creative_ideas(prompt)
                
                output = f"🧠 Creative Ideas for: {prompt}\n"
                output += "=" * 50 + "\n"
                
                # Show idea variations
                if result.get('idea_variations'):
                    output += "💡 IDEA VARIATIONS:\n"
                    for i, variation in enumerate(result['idea_variations'][:3], 1):
                        output += f"{i}. {variation}\n"
                    output += "\n"
                
                # Show top ranked ideas
                if result.get('ranked_ideas'):
                    output += "🏆 TOP CREATIVE IDEAS:\n"
                    for i, idea in enumerate(result['ranked_ideas'][:3], 1):
                        output += f"{i}. {idea['idea']}\n"
                        output += f"   Creativity: {idea['creativity_score']:.2f}, Feasibility: {idea['feasibility_score']:.2f}\n\n"
                
                # Show novel angles
                if result.get('novel_angles'):
                    output += "🎯 NOVEL ANGLES:\n"
                    for angle in result['novel_angles'][:2]:
                        output += f"• {angle}\n"
                
                return output
                
            except Exception as e:
                return f"Creative idea exploration failed: {str(e)}"
        
        def analyze_webpage_visually_tool(url: str) -> str:
            """Analyze webpage with visual processing and enhancement"""
            try:
                result = self.diffusion_agent.analyze_webpage_visually(url, "content")
                
                if 'error' in result:
                    return f"Visual analysis failed: {result['error']}"
                
                output = f"👁️ Visual Analysis of: {url}\n"
                output += "=" * 50 + "\n"
                
                # Show insights
                if result.get('insights'):
                    output += "🔍 VISUAL INSIGHTS:\n"
                    for insight in result['insights']:
                        output += f"• {insight}\n"
                    output += "\n"
                
                # Show visual analysis details
                if result.get('visual_analysis'):
                    analysis = result['visual_analysis']
                    output += "📊 ANALYSIS DETAILS:\n"
                    output += f"• Image dimensions: {analysis.get('image_dimensions', 'N/A')}\n"
                    output += f"• Visual complexity: {analysis.get('visual_complexity', 0):.2f}\n"
                    output += f"• Text regions detected: {len(analysis.get('text_regions', []))}\n"
                    
                    if analysis.get('dominant_colors'):
                        output += f"• Dominant colors: {', '.join(analysis['dominant_colors'][:3])}\n"
                
                output += f"\n📷 Enhanced image available in processing results"
                
                return output
                
            except Exception as e:
                return f"Visual webpage analysis failed: {str(e)}"
        
        def generate_diverse_research_plan_tool(research_question: str) -> str:
            """Generate diverse research plan using diffusion sampling"""
            try:
                result = self.diffusion_agent.generate_diverse_research_plan(research_question)
                
                output = f"📋 Diverse Research Plan for: {research_question}\n"
                output += "=" * 50 + "\n"
                
                selected_plan = result.get('selected_plan', {})
                plan_steps = selected_plan.get('plan_steps', [])
                
                output += "🎯 SELECTED RESEARCH PLAN:\n"
                for step in plan_steps:
                    output += f"{step['step_id']}. {step['description']}\n"
                    if step.get('tools_suggested'):
                        output += f"   Tools: {', '.join(step['tools_suggested'])}\n"
                    output += f"   Effort: {step.get('estimated_effort', 0):.2f}\n\n"
                
                # Show planning confidence
                confidence = result.get('planning_confidence', 0)
                output += f"🎯 Planning Confidence: {confidence:.2f}\n"
                
                # Show alternative plans
                alt_plans = result.get('alternative_plans', [])
                if alt_plans:
                    output += f"\n🔄 {len(alt_plans)} alternative plans generated\n"
                
                return output
                
            except Exception as e:
                return f"Diverse research plan generation failed: {str(e)}"
        
        def generate_synthetic_data_tool(topic: str) -> str:
            """Generate synthetic research data for a topic"""
            try:
                # Create sample CSV data for demonstration
                sample_data = pd.DataFrame({
                    'category': [f'{topic}_aspect_{i}' for i in range(1, 6)],
                    'value': [10, 25, 15, 30, 20],
                    'importance': [0.8, 0.9, 0.7, 0.95, 0.75]
                })
                
                charts = self.diffusion_agent.generate_multimodal_content(sample_data)
                
                output = f"📊 Synthetic Data Generated for: {topic}\n"
                output += "=" * 50 + "\n"
                
                output += f"Generated {len(charts)} synthetic visualizations:\n"
                for chart in charts:
                    output += f"• {chart['type'].title()} Chart: {chart['description']}\n"
                
                output += f"\n💾 Charts available in base64 format for display"
                
                return output
                
            except Exception as e:
                return f"Synthetic data generation failed: {str(e)}"
        
        def generate_aligned_content_tool(prompt: str) -> str:
            """Generate content aligned with human preferences"""
            try:
                result = self.diffusion_agent.generate_aligned_content(
                    prompt, style_preference="neutral", quality_preference="high"
                )
                
                output = f"✨ Aligned Content for: {prompt}\n"
                output += "=" * 50 + "\n"
                
                output += f"Generated Text:\n{result.get('generated_text', 'No content generated')}\n\n"
                output += f"Alignment Score: {result.get('alignment_score', 0):.2f}\n"
                output += f"Generation Method: {result.get('generation_method', 'unknown')}\n"
                
                return output
                
            except Exception as e:
                return f"Aligned content generation failed: {str(e)}"
        
        return [
            Tool(
                name="explore_creative_ideas",
                description="Explore creative research ideas and novel angles using diffusion-based brainstorming. Input your research topic or question.",
                func=explore_creative_ideas_tool
            ),
            Tool(
                name="analyze_webpage_visually",
                description="Analyze webpage with visual processing, enhancement, and content highlighting. Input the URL to analyze.",
                func=analyze_webpage_visually_tool
            ),
            Tool(
                name="generate_diverse_research_plan",
                description="Generate diverse, creative research plans using diffusion sampling. Input your research question.",
                func=generate_diverse_research_plan_tool
            ),
            Tool(
                name="generate_synthetic_data",
                description="Generate synthetic research data and visualizations for a topic. Input the research topic.",
                func=generate_synthetic_data_tool
            ),
            Tool(
                name="generate_aligned_content",
                description="Generate content aligned with human preferences for quality and style. Input your content prompt.",
                func=generate_aligned_content_tool
            )
        ]
    
    def get_tool_usage_guide(self) -> str:
        """Get comprehensive guide for using research tools"""
        guide = """
🛠️ Research Tools Arsenal - Usage Guide
=====================================

📡 WEB RESEARCH TOOLS:
• web_search - General web search using DuckDuckGo
• wikipedia_search - Encyclopedic information and definitions
• arxiv_search - Academic papers and research
• news_search - Recent news and current events
• scrape_webpage - Detailed analysis of specific webpages

📄 DOCUMENT PROCESSING TOOLS:
• process_pdf_url - Download and analyze PDFs from URLs
• process_local_pdf - Analyze local PDF files
• analyze_document_structure - Extract sections and structure from text

📊 DATA VISUALIZATION TOOLS:
• create_timeline_visualization - Timeline charts from chronological data
• create_concept_network - Network graphs of concept relationships
• create_metrics_dashboard - Comprehensive metrics dashboards
• create_word_frequency_chart - Word frequency analysis
• analyze_research_data - Suggest appropriate visualizations

� DIFFUSSION-ENHANCED TOOLS:
• explore_creative_ideas - AI-powered creative brainstorming and idea generation
• analyze_webpage_visually - Visual webpage analysis with content highlighting
• generate_diverse_research_plan - Creative research planning with multiple approaches
• generate_synthetic_data - Create synthetic data and visualizations
• generate_aligned_content - Generate human-aligned, high-quality content

🎯 TOOL SELECTION TIPS:
• Start with web_search for general information
• Use wikipedia_search for background and definitions
• Use arxiv_search for academic/scientific topics
• Use news_search for current events and recent developments
• Use document tools when you have specific PDFs to analyze
• Use visualization tools to create charts and graphs from your findings

🔄 WORKFLOW SUGGESTIONS:
1. Begin with web_search or wikipedia_search for overview
2. Use explore_creative_ideas for brainstorming and novel angles
3. Use generate_diverse_research_plan for structured approach
4. Use arxiv_search for academic depth
5. Process relevant PDFs with document tools
6. Use analyze_webpage_visually for visual content analysis
7. Create visualizations to present findings
8. Use generate_aligned_content for high-quality summaries
9. Use news_search for latest developments

🌊 DIFFUSION WORKFLOW:
• Start with explore_creative_ideas for creative research angles
• Use generate_diverse_research_plan for comprehensive planning
• Apply analyze_webpage_visually for visual content analysis
• Generate synthetic_data for data augmentation
• Use generate_aligned_content for polished outputs
"""
        return guide

def get_research_tools_manager():
    """Get the research tools manager instance"""
    return ResearchToolsManager()

def get_all_research_tools():
    """Get all Phase 3 research tools"""
    manager = ResearchToolsManager()
    
    # Add tool suggestion capability
    def suggest_tools_tool(query: str) -> str:
        """Suggest appropriate research tools for a query"""
        try:
            suggestions = manager.suggest_tools_for_query(query)
            
            output = f"Tool Suggestions for: '{query}'\n"
            output += "=" * 50 + "\n"
            
            if suggestions['recommended']:
                output += "🎯 RECOMMENDED TOOLS:\n"
                for tool in suggestions['recommended']:
                    output += f"  • {tool}\n"
                output += "\n"
            
            if suggestions['optional']:
                output += "🔧 OPTIONAL TOOLS:\n"
                for tool in suggestions['optional']:
                    output += f"  • {tool}\n"
                output += "\n"
            
            if suggestions['visualization']:
                output += "📊 VISUALIZATION TOOLS:\n"
                for tool in suggestions['visualization']:
                    output += f"  • {tool}\n"
                output += "\n"
            
            output += "💡 Use 'get_tools_guide' for detailed usage instructions"
            
            return output
            
        except Exception as e:
            return f"Tool suggestion failed: {str(e)}"
    
    def get_tools_guide_tool(dummy_input: str = "") -> str:
        """Get comprehensive research tools usage guide"""
        return manager.get_tool_usage_guide()
    
    # Add the management tools
    management_tools = [
        Tool(
            name="suggest_research_tools",
            description="Suggest appropriate research tools based on your research query. Input your research question or topic.",
            func=suggest_tools_tool
        ),
        Tool(
            name="get_tools_guide",
            description="Get comprehensive guide for using all research tools. No input required.",
            func=get_tools_guide_tool
        )
    ]
    
    return manager.get_all_tools() + management_tools