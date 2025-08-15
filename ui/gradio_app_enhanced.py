# -*- coding: utf-8 -*-
"""
Enhanced Gradio Web Interface for AI Research Agent with Diffusion Capabilities
Alternative web interface with focus on simplicity, sharing, and diffusion features
"""

import gradio as gr
import json
import time
import base64
import io
from datetime import datetime
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import our research agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.research_agent import create_agent
from memory.advanced_memory_manager import AdvancedMemoryManager

# Diffusion integration
try:
    from diffusion.research_agent_integration import create_diffusion_enhanced_agent
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False

class EnhancedGradioResearchInterface:
    """Enhanced Gradio interface with diffusion capabilities"""
    
    def __init__(self):
        self.agent = None
        self.memory_manager = AdvancedMemoryManager(enable_diffusion=True)
        self.research_history = []
        
        # Initialize diffusion agent if available
        self.diffusion_agent = None
        self.diffusion_enabled = False
        
        if DIFFUSION_AVAILABLE:
            try:
                self.diffusion_agent = create_diffusion_enhanced_agent(enable_all=True)
                self.diffusion_enabled = True
                print("✅ Diffusion capabilities enabled")
            except Exception as e:
                print(f"⚠️ Diffusion agent initialization failed: {e}")
                self.diffusion_agent = None
                self.diffusion_enabled = False
        else:
            print("⚠️ Diffusion capabilities not available")
    
    def initialize_agent(self):
        """Initialize the research agent"""
        if self.agent is None:
            self.agent = create_agent()
        return self.agent
    
    def conduct_research(self, question: str, enable_hypothesis: bool = True, 
                        enable_multi_agent: bool = True, use_diffusion: bool = True) -> Tuple[str, str, str, str]:
        """Conduct research with optional diffusion enhancement"""
        
        if not question.strip():
            return "Please enter a research question.", "", "", ""
        
        try:
            # Initialize agent
            agent = self.initialize_agent()
            
            # Prepare initial state with diffusion components
            initial_state = {
                "messages": [],
                "research_question": question,
                "research_plan": [],
                "current_step": 0,
                "findings": [],
                "final_answer": "",
                "iteration_count": 0,
                "hypotheses": [],
                "multi_agent_analysis": {},
                "quality_assessment": {},
                "intelligence_insights": {},
                # Diffusion components
                "diffusion_enhanced": use_diffusion and self.diffusion_enabled,
                "synthetic_contexts": [],
                "visual_analysis_results": {},
                "creative_ideas": {}
            }
            
            # Execute research
            result = agent.invoke(initial_state)
            
            # Store in history
            research_record = {
                'question': question,
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'quality_score': result.get('quality_assessment', {}).get('overall_quality_score', 'N/A'),
                'diffusion_enhanced': use_diffusion and self.diffusion_enabled
            }
            self.research_history.append(research_record)
            
            # Format results for display
            final_answer = result.get("final_answer", "No final answer generated")
            
            # Add diffusion enhancement notice
            if use_diffusion and self.diffusion_enabled:
                final_answer = "🌊 **Diffusion-Enhanced Research Results**\n\n" + final_answer
            
            # Research process summary
            research_plan = result.get("research_plan", [])
            findings = result.get("findings", [])
            
            process_summary = "## Research Process\n\n"
            if use_diffusion and self.diffusion_enabled:
                process_summary += "🌊 **Diffusion-Enhanced Planning Applied**\n\n"
            
            process_summary += "### Research Plan:\n"
            for i, step in enumerate(research_plan, 1):
                process_summary += f"{i}. {step}\n"
            
            process_summary += f"\n### Findings Summary:\n"
            process_summary += f"- Total research steps completed: {len(findings)}\n"
            process_summary += f"- External sources consulted: {sum(1 for f in findings if f.get('external_research'))}\n"
            
            # Add diffusion-specific metrics
            if use_diffusion and self.diffusion_enabled:
                synthetic_contexts = result.get("synthetic_contexts", [])
                if synthetic_contexts:
                    process_summary += f"- Synthetic contexts generated: {len(synthetic_contexts)}\n"
            
            # Intelligence analysis summary
            intelligence_summary = "## Intelligence Analysis\n\n"
            
            # Multi-agent analysis
            multi_agent_analysis = result.get("multi_agent_analysis", {})
            if multi_agent_analysis:
                confidence_scores = multi_agent_analysis.get("confidence_scores", {})
                intelligence_summary += "### Multi-Agent Collaboration:\n"
                intelligence_summary += f"- Researcher Confidence: {confidence_scores.get('researcher_avg', 0):.2f}\n"
                intelligence_summary += f"- Critic Confidence: {confidence_scores.get('critic_avg', 0):.2f}\n"
                intelligence_summary += f"- Synthesizer Confidence: {confidence_scores.get('synthesis_confidence', 0):.2f}\n"
            
            # Hypotheses
            hypotheses = result.get("hypotheses", [])
            if hypotheses:
                intelligence_summary += "\n### Generated Hypotheses:\n"
                for i, hyp in enumerate(hypotheses, 1):
                    intelligence_summary += f"{i}. **{hyp['statement']}**\n"
                    intelligence_summary += f"   - Type: {hyp['type']}\n"
                    intelligence_summary += f"   - Confidence: {hyp['confidence']:.2f}\n\n"
            
            # Add creative ideas if available
            creative_ideas = result.get("creative_ideas", {})
            if creative_ideas and use_diffusion:
                intelligence_summary += "\n### 🧠 Creative Research Angles:\n"
                for idea in creative_ideas.get('ranked_ideas', [])[:3]:
                    intelligence_summary += f"- **{idea.get('idea', 'N/A')}**\n"
                    intelligence_summary += f"  Creativity: {idea.get('creativity_score', 0):.2f}, "
                    intelligence_summary += f"Feasibility: {idea.get('feasibility_score', 0):.2f}\n\n"
            
            # Quality assessment
            quality_assessment = result.get("quality_assessment", {})
            quality_summary = "## Quality Assessment\n\n"
            
            if quality_assessment:
                quality_summary += f"- **Overall Quality Score:** {quality_assessment.get('overall_quality_score', 'N/A')}/10\n"
                quality_summary += f"- **Confidence Level:** {quality_assessment.get('confidence_assessment', 'N/A'):.2f}\n"
                quality_summary += f"- **Total Findings:** {quality_assessment.get('total_findings', 0)}\n"
                quality_summary += f"- **External Sources Used:** {quality_assessment.get('external_sources_used', 0)}\n"
                quality_summary += f"- **Source Diversity:** {quality_assessment.get('source_diversity', 0)}\n"
                
                if use_diffusion and self.diffusion_enabled:
                    quality_summary += f"- **Diffusion Enhancement:** ✅ Applied\n"
                
                quality_indicators = quality_assessment.get("quality_indicators", {})
                quality_summary += "\n### Quality Indicators:\n"
                for indicator, status in quality_indicators.items():
                    icon = "✅" if status else "❌"
                    quality_summary += f"{icon} {indicator.replace('_', ' ').title()}\n"
            
            return final_answer, process_summary, intelligence_summary, quality_summary
            
        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            return error_msg, "", "", ""
    
    # Diffusion-specific methods
    def explore_creative_ideas(self, topic: str, creativity_level: float = 0.8) -> str:
        """Explore creative ideas using diffusion"""
        if not self.diffusion_enabled:
            return "❌ Diffusion capabilities not available"
        
        if not topic.strip():
            return "Please enter a research topic"
        
        try:
            result = self.diffusion_agent.explore_creative_ideas(topic)
            
            output = f"# 🧠 Creative Ideas for: {topic}\n\n"
            
            # Show idea variations
            if result.get('idea_variations'):
                output += "## 💡 Idea Variations\n"
                for i, variation in enumerate(result['idea_variations'][:3], 1):
                    output += f"{i}. {variation}\n"
                output += "\n"
            
            # Show top ranked ideas
            if result.get('ranked_ideas'):
                output += "## 🏆 Top Creative Ideas\n"
                for i, idea in enumerate(result['ranked_ideas'][:5], 1):
                    output += f"**{i}. {idea['idea']}**\n"
                    output += f"   - Creativity: {idea['creativity_score']:.2f}\n"
                    output += f"   - Feasibility: {idea['feasibility_score']:.2f}\n\n"
            
            # Show novel angles
            if result.get('novel_angles'):
                output += "## 🎯 Novel Research Angles\n"
                for angle in result['novel_angles'][:3]:
                    output += f"- {angle}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Creative idea exploration failed: {e}"
    
    def analyze_webpage_visually(self, url: str, focus: str = "content") -> Tuple[str, str]:
        """Analyze webpage visually using diffusion"""
        if not self.diffusion_enabled:
            return "❌ Diffusion capabilities not available", ""
        
        if not url.strip():
            return "Please enter a valid URL", ""
        
        try:
            result = self.diffusion_agent.analyze_webpage_visually(url, focus)
            
            if 'error' in result:
                return f"❌ Analysis failed: {result['error']}", ""
            
            output = f"# 👁️ Visual Analysis of: {url}\n\n"
            output += f"**Analysis Focus:** {focus.title()}\n\n"
            
            # Show insights
            if result.get('insights'):
                output += "## 🔍 Visual Insights\n"
                for insight in result['insights']:
                    output += f"- {insight}\n"
                output += "\n"
            
            # Show analysis details
            if result.get('visual_analysis'):
                analysis = result['visual_analysis']
                output += "## 📊 Analysis Details\n"
                output += f"- **Image dimensions:** {analysis.get('image_dimensions', 'N/A')}\n"
                output += f"- **Visual complexity:** {analysis.get('visual_complexity', 0):.2f}\n"
                output += f"- **Text regions detected:** {len(analysis.get('text_regions', []))}\n"
                
                if analysis.get('dominant_colors'):
                    output += f"- **Dominant colors:** {', '.join(analysis['dominant_colors'][:3])}\n"
            
            # Return enhanced image if available
            enhanced_image = result.get('enhanced_image', '')
            
            return output, enhanced_image
            
        except Exception as e:
            return f"❌ Visual analysis failed: {e}", ""
    
    def generate_research_plan(self, question: str, creativity_boost: float = 0.2) -> str:
        """Generate diverse research plan using diffusion"""
        if not self.diffusion_enabled:
            return "❌ Diffusion capabilities not available"
        
        if not question.strip():
            return "Please enter a research question"
        
        try:
            result = self.diffusion_agent.generate_diverse_research_plan(question)
            
            output = f"# 📋 Diverse Research Plan\n\n"
            output += f"**Question:** {question}\n\n"
            
            # Show selected plan
            selected_plan = result.get('selected_plan', {})
            if selected_plan:
                output += "## 🎯 Selected Research Plan\n"
                
                plan_steps = selected_plan.get('plan_steps', [])
                for step in plan_steps:
                    output += f"**{step['step_id']}. {step['description']}**\n"
                    if step.get('tools_suggested'):
                        output += f"   - Tools: {', '.join(step['tools_suggested'])}\n"
                    output += f"   - Effort: {step.get('estimated_effort', 0):.2f}\n\n"
                
                # Show planning metrics
                output += "## 📊 Planning Metrics\n"
                output += f"- **Total Steps:** {len(plan_steps)}\n"
                output += f"- **Planning Confidence:** {result.get('planning_confidence', 0):.2f}\n"
                complexity = selected_plan.get('estimated_complexity', 0)
                output += f"- **Complexity:** {complexity:.2f}\n"
            
            # Show alternative plans
            alt_plans = result.get('alternative_plans', [])
            if alt_plans:
                output += f"\n## 🔄 Alternative Plans ({len(alt_plans)} generated)\n"
                for i, alt_plan in enumerate(alt_plans[:2], 1):
                    output += f"### Alternative Plan {i}\n"
                    alt_steps = alt_plan.get('plan_steps', [])
                    for step in alt_steps[:3]:  # Show first 3 steps
                        output += f"- {step['description']}\n"
                    if len(alt_steps) > 3:
                        output += f"... and {len(alt_steps) - 3} more steps\n"
                    output += "\n"
            
            return output
            
        except Exception as e:
            return f"❌ Research planning failed: {e}"
    
    def generate_synthetic_data(self, topic: str, chart_types: List[str]) -> Tuple[str, str]:
        """Generate synthetic data and visualizations"""
        if not self.diffusion_enabled:
            return "❌ Diffusion capabilities not available", ""
        
        if not topic.strip():
            return "Please enter a data topic", ""
        
        try:
            # Create sample data
            data_points = 10
            sample_data = pd.DataFrame({
                'category': [f'{topic}_aspect_{i}' for i in range(1, data_points + 1)],
                'value': np.random.randint(10, 100, data_points),
                'importance': np.random.uniform(0.3, 1.0, data_points),
                'trend': np.random.choice(['increasing', 'decreasing', 'stable'], data_points)
            })
            
            # Generate synthetic visualizations
            charts = self.diffusion_agent.generate_multimodal_content(sample_data)
            
            output = f"# 📊 Synthetic Data for: {topic}\n\n"
            output += "## 📈 Generated Data\n"
            output += sample_data.to_markdown(index=False)
            output += f"\n\n## 🎨 Synthetic Visualizations\n"
            output += f"Generated {len(charts)} synthetic visualizations:\n"
            
            for chart in charts:
                output += f"- **{chart['type'].title()} Chart:** {chart['description']}\n"
            
            # Return first chart image if available
            chart_image = ""
            if charts and charts[0].get('data'):
                chart_image = charts[0]['data']
            
            return output, chart_image
            
        except Exception as e:
            return f"❌ Synthetic data generation failed: {e}", ""
    
    def generate_aligned_content(self, prompt: str, style: str = "neutral", quality: str = "high") -> str:
        """Generate human-aligned content"""
        if not self.diffusion_enabled:
            return "❌ Diffusion capabilities not available"
        
        if not prompt.strip():
            return "Please enter a content prompt"
        
        try:
            result = self.diffusion_agent.generate_aligned_content(prompt, style, quality)
            
            output = f"# ✨ Aligned Content\n\n"
            output += f"**Prompt:** {prompt}\n"
            output += f"**Style:** {style.title()}\n"
            output += f"**Quality:** {quality.title()}\n\n"
            
            output += "## 📝 Generated Content\n"
            output += result.get('generated_text', 'No content generated')
            
            output += f"\n\n## 📊 Alignment Metrics\n"
            output += f"- **Alignment Score:** {result.get('alignment_score', 0):.2f}\n"
            output += f"- **Generation Method:** {result.get('generation_method', 'unknown')}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Aligned content generation failed: {e}"
    
    def get_diffusion_stats(self) -> str:
        """Get diffusion system statistics"""
        if not self.diffusion_enabled:
            return "❌ Diffusion capabilities not available"
        
        try:
            stats = self.diffusion_agent.get_comprehensive_stats()
            
            output = "# 🌊 Diffusion System Statistics\n\n"
            
            # Core stats
            core_stats = stats['core_diffusion']
            output += "## 🔧 Core Diffusion\n"
            output += f"- **Model Trained:** {'✅' if core_stats['model_trained'] else '❌'}\n"
            output += f"- **Model Dimension:** {core_stats['model_dim']}\n"
            output += f"- **Timesteps:** {core_stats['num_timesteps']}\n"
            output += f"- **Training Contexts:** {stats['training_contexts_count']}\n\n"
            
            # Capabilities
            capabilities = stats['capabilities']
            output += "## 🌊 Capabilities\n"
            output += f"- **Synthetic Data Generation:** {'✅' if capabilities['synthetic_data_generation'] else '❌'}\n"
            output += f"- **Denoising Layer:** {'✅' if capabilities['denoising_layer'] else '❌'}\n"
            output += f"- **Planning Diffusion:** {'✅' if capabilities['planning_diffusion'] else '❌'}\n"
            output += f"- **Vision & Creativity:** {'✅' if capabilities['vision_creativity'] else '❌'}\n"
            output += f"- **RLHF Integration:** {'✅' if capabilities['rlhf_integration'] else '❌'}\n"
            
            return output
            
        except Exception as e:
            return f"❌ Failed to get diffusion stats: {e}"
    
    # Original methods (updated)
    def get_research_suggestions(self) -> str:
        """Get research question suggestions"""
        suggestions = [
            "What are the ethical implications of AI in healthcare?",
            "How do different renewable energy technologies compare in efficiency?",
            "What are the latest breakthroughs in quantum computing?",
            "How does climate change affect global food security?",
            "What are the emerging trends in cybersecurity?",
            "How do different economic models predict inflation?",
            "What are the competing theories about consciousness?",
            "How does social media impact mental health?",
            "What are the potential applications of CRISPR gene editing?",
            "How do neural networks learn and make decisions?"
        ]
        
        suggestions_text = "## 💡 Research Question Suggestions\n\n"
        for i, suggestion in enumerate(suggestions, 1):
            suggestions_text += f"{i}. {suggestion}\n"
        
        return suggestions_text
    
    def get_memory_statistics(self) -> str:
        """Get memory system statistics"""
        try:
            stats = self.memory_manager.hierarchical_memory.get_memory_statistics()
            
            stats_text = "# 🧠 Memory System Statistics\n\n"
            stats_text += f"- **Short-term Memory:** {stats.get('short_term_count', 0)} items\n"
            stats_text += f"- **Long-term Memory:** {stats.get('long_term_count', 0)} items\n"
            stats_text += f"- **Episodic Memory:** {stats.get('episodic_count', 0)} episodes\n"
            stats_text += f"- **Knowledge Graph Nodes:** {stats.get('knowledge_graph_nodes', 0)}\n"
            stats_text += f"- **Knowledge Graph Edges:** {stats.get('knowledge_graph_edges', 0)}\n"
            stats_text += f"- **Concepts Tracked:** {stats.get('concepts_tracked', 0)}\n"
            stats_text += f"- **Citations Tracked:** {stats.get('citations_tracked', 0)}\n"
            
            # Add diffusion-specific memory stats
            if self.diffusion_enabled:
                try:
                    diffusion_stats = self.memory_manager.get_diffusion_memory_stats()
                    stats_text += f"\n## 🌊 Diffusion Memory Enhancement\n"
                    stats_text += f"- **Diffusion Enabled:** {'✅' if diffusion_stats['diffusion_enabled'] else '❌'}\n"
                    stats_text += f"- **Synthetic Contexts Cached:** {diffusion_stats['synthetic_contexts_cached']}\n"
                except:
                    pass
            
            return stats_text
            
        except Exception as e:
            return f"❌ Error retrieving memory statistics: {e}"
    
    def get_research_history(self) -> str:
        """Get research history summary"""
        if not self.research_history:
            return "📚 No research history available."
        
        history_text = "# 📚 Recent Research History\n\n"
        
        for i, research in enumerate(self.research_history[-5:], 1):
            history_text += f"## Research {i}\n"
            history_text += f"- **Question:** {research['question']}\n"
            history_text += f"- **Quality Score:** {research.get('quality_score', 'N/A')}\n"
            history_text += f"- **Timestamp:** {research.get('timestamp', 'Unknown')}\n"
            if research.get('diffusion_enhanced'):
                history_text += f"- **Diffusion Enhanced:** ✅\n"
            history_text += "\n"
        
        return history_text
    
    def create_interface(self):
        """Create the enhanced Gradio interface with diffusion capabilities"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .diffusion-enabled {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="AI Research Agent - Diffusion Enhanced") as interface:
            
            # Header
            if self.diffusion_enabled:
                gr.HTML("""
                <div class="diffusion-enabled">
                    <h2>🌊 Diffusion-Enhanced AI Research Agent</h2>
                    <p>The Most Advanced AI Research Intelligence System with Diffusion Models</p>
                </div>
                """)
            else:
                gr.Markdown("""
                # 🔬 AI Research Agent
                ### The Most Advanced AI Research Intelligence System
                """)
            
            # Main research tab
            with gr.Tab("🔍 Research"):
                with gr.Row():
                    with gr.Column(scale=2):
                        research_question = gr.Textbox(
                            label="Research Question",
                            placeholder="Enter your research question here...",
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            enable_hypothesis = gr.Checkbox(
                                label="Enable Hypothesis Generation",
                                value=True
                            )
                            enable_multi_agent = gr.Checkbox(
                                label="Enable Multi-Agent Analysis",
                                value=True
                            )
                            if self.diffusion_enabled:
                                use_diffusion = gr.Checkbox(
                                    label="🌊 Use Diffusion Enhancement",
                                    value=True
                                )
                        
                        with gr.Row():
                            research_btn = gr.Button("🚀 Start Research", variant="primary")
                            suggestions_btn = gr.Button("💡 Get Suggestions")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛠️ Research Tools Available")
                        tools_text = """
                        **Web Research:**
                        • DuckDuckGo Search
                        • Wikipedia Integration
                        • arXiv Academic Papers
                        • News Search
                        
                        **Intelligence Layer:**
                        • Multi-Agent Analysis
                        • Hypothesis Testing
                        • Quality Assessment
                        """
                        
                        if self.diffusion_enabled:
                            tools_text += """
                        
                        **🌊 Diffusion Enhanced:**
                        • Creative Idea Generation
                        • Visual Webpage Analysis
                        • Diverse Planning
                        • Synthetic Data Creation
                        • Human-Aligned Content
                        """
                        
                        gr.Markdown(tools_text)
                
                # Research suggestions
                suggestions_output = gr.Markdown(visible=False)
                
                # Results tabs
                with gr.Tab("📝 Final Answer"):
                    final_answer_output = gr.Markdown()
                
                with gr.Tab("🔬 Research Process"):
                    process_output = gr.Markdown()
                
                with gr.Tab("🧠 Intelligence Analysis"):
                    intelligence_output = gr.Markdown()
                
                with gr.Tab("📊 Quality Assessment"):
                    quality_output = gr.Markdown()
            
            # Diffusion-specific tabs
            if self.diffusion_enabled:
                with gr.Tab("🧠 Creative Ideas"):
                    with gr.Row():
                        with gr.Column():
                            idea_topic = gr.Textbox(
                                label="Research Topic",
                                placeholder="e.g., sustainable energy solutions"
                            )
                            creativity_level = gr.Slider(
                                label="Creativity Level",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.1
                            )
                            ideas_btn = gr.Button("🚀 Generate Creative Ideas", variant="primary")
                        
                        with gr.Column():
                            ideas_output = gr.Markdown()
                
                with gr.Tab("👁️ Visual Analysis"):
                    with gr.Row():
                        with gr.Column():
                            url_input = gr.Textbox(
                                label="Webpage URL",
                                placeholder="https://example.com"
                            )
                            analysis_focus = gr.Dropdown(
                                label="Analysis Focus",
                                choices=["content", "headlines", "navigation", "layout"],
                                value="content"
                            )
                            visual_btn = gr.Button("🔍 Analyze Webpage", variant="primary")
                        
                        with gr.Column():
                            visual_output = gr.Markdown()
                            enhanced_image = gr.Image(label="Enhanced Screenshot", visible=False)
                
                with gr.Tab("📋 Smart Planning"):
                    with gr.Row():
                        with gr.Column():
                            planning_question = gr.Textbox(
                                label="Research Question",
                                placeholder="What are the environmental impacts of AI?",
                                lines=3
                            )
                            creativity_boost = gr.Slider(
                                label="Creativity Boost",
                                minimum=0.0,
                                maximum=0.5,
                                value=0.2,
                                step=0.1
                            )
                            planning_btn = gr.Button("🎯 Generate Research Plan", variant="primary")
                        
                        with gr.Column():
                            planning_output = gr.Markdown()
                
                with gr.Tab("📊 Synthetic Data"):
                    with gr.Row():
                        with gr.Column():
                            data_topic = gr.Textbox(
                                label="Data Topic",
                                placeholder="e.g., climate change trends"
                            )
                            chart_types = gr.CheckboxGroup(
                                label="Chart Types",
                                choices=["bar", "line", "scatter", "histogram", "heatmap"],
                                value=["bar", "line"]
                            )
                            synthetic_btn = gr.Button("📊 Generate Synthetic Data", variant="primary")
                        
                        with gr.Column():
                            synthetic_output = gr.Markdown()
                            synthetic_image = gr.Image(label="Generated Chart", visible=False)
                
                with gr.Tab("✨ Aligned Content"):
                    with gr.Row():
                        with gr.Column():
                            content_prompt = gr.Textbox(
                                label="Content Prompt",
                                placeholder="Explain the benefits of renewable energy",
                                lines=3
                            )
                            style_preference = gr.Dropdown(
                                label="Style Preference",
                                choices=["concise", "neutral", "detailed", "creative"],
                                value="neutral"
                            )
                            quality_preference = gr.Dropdown(
                                label="Quality Preference",
                                choices=["low", "medium", "high"],
                                value="high"
                            )
                            aligned_btn = gr.Button("✨ Generate Aligned Content", variant="primary")
                        
                        with gr.Column():
                            aligned_output = gr.Markdown()
            
            # Memory & Statistics tab
            with gr.Tab("📊 Memory & Statistics"):
                with gr.Row():
                    with gr.Column():
                        memory_btn = gr.Button("🧠 Get Memory Statistics")
                        memory_output = gr.Markdown()
                    
                    with gr.Column():
                        history_btn = gr.Button("📚 Get Research History")
                        history_output = gr.Markdown()
                
                if self.diffusion_enabled:
                    with gr.Row():
                        with gr.Column():
                            diffusion_stats_btn = gr.Button("🌊 Get Diffusion Statistics")
                            diffusion_stats_output = gr.Markdown()
            
            # About tab
            with gr.Tab("ℹ️ About"):
                about_text = """
                ## About AI Research Agent
                
                This is the most advanced AI research intelligence system ever built, featuring:
                """
                
                if self.diffusion_enabled:
                    about_text += """
                ### 🌊 Diffusion Enhancement (Phase 8)
                - **Synthetic Data Generation**: Context augmentation and balanced datasets
                - **Denoising Layer**: Noise-robust retrieval and reasoning
                - **Planning Diffusion**: Diverse plan generation and trajectory refinement
                - **Vision & Creativity**: Visual analysis and creative brainstorming
                - **RLHF Integration**: Classifier-free guidance and adversarial training
                """
                
                about_text += """
                ### 🧠 Intelligence Layer (Phase 4)
                - **Multi-Agent Collaboration**: Researcher, Critic, and Synthesizer agents
                - **Hypothesis Generation**: Automatic creation of testable hypotheses
                - **Quality Assessment**: Comprehensive research validation
                
                ### 🔬 Research Tools Arsenal (Phase 3)
                - **Web Research Suite**: DuckDuckGo, Wikipedia, arXiv, News
                - **Document Processing**: PDF analysis and structure extraction
                - **Data Visualization**: Charts, networks, and dashboards
                
                ### 🧠 Advanced Memory System (Phase 2)
                - **Hierarchical Memory**: Short-term, long-term, and episodic
                - **Knowledge Graphs**: Automatic concept relationship mapping
                - **Citation Tracking**: Source credibility and network analysis
                
                ### 🎯 Core Capabilities (Phase 1)
                - **Structured Research**: Multi-step planning and execution
                - **ReAct Pattern**: Reasoning and acting in structured loops
                - **Memory Integration**: Persistent knowledge across sessions
                
                ---
                
                **Built with:** LangGraph • LangMem • Groq • NetworkX • Plotly • Diffusion Models
                """
                
                gr.Markdown(about_text)
            
            # Event handlers
            def conduct_research_wrapper(*args):
                if self.diffusion_enabled and len(args) >= 4:
                    return self.conduct_research(args[0], args[1], args[2], args[3])
                else:
                    return self.conduct_research(args[0], args[1], args[2], True)
            
            # Connect main research events
            if self.diffusion_enabled:
                research_btn.click(
                    fn=conduct_research_wrapper,
                    inputs=[research_question, enable_hypothesis, enable_multi_agent, use_diffusion],
                    outputs=[final_answer_output, process_output, intelligence_output, quality_output]
                )
            else:
                research_btn.click(
                    fn=self.conduct_research,
                    inputs=[research_question, enable_hypothesis, enable_multi_agent],
                    outputs=[final_answer_output, process_output, intelligence_output, quality_output]
                )
            
            suggestions_btn.click(
                fn=self.get_research_suggestions,
                outputs=[suggestions_output]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[suggestions_output]
            )
            
            # Connect diffusion events
            if self.diffusion_enabled:
                ideas_btn.click(
                    fn=self.explore_creative_ideas,
                    inputs=[idea_topic, creativity_level],
                    outputs=[ideas_output]
                )
                
                visual_btn.click(
                    fn=self.analyze_webpage_visually,
                    inputs=[url_input, analysis_focus],
                    outputs=[visual_output, enhanced_image]
                ).then(
                    fn=lambda img: gr.update(visible=bool(img)),
                    inputs=[enhanced_image],
                    outputs=[enhanced_image]
                )
                
                planning_btn.click(
                    fn=self.generate_research_plan,
                    inputs=[planning_question, creativity_boost],
                    outputs=[planning_output]
                )
                
                synthetic_btn.click(
                    fn=self.generate_synthetic_data,
                    inputs=[data_topic, chart_types],
                    outputs=[synthetic_output, synthetic_image]
                ).then(
                    fn=lambda img: gr.update(visible=bool(img)),
                    inputs=[synthetic_image],
                    outputs=[synthetic_image]
                )
                
                aligned_btn.click(
                    fn=self.generate_aligned_content,
                    inputs=[content_prompt, style_preference, quality_preference],
                    outputs=[aligned_output]
                )
                
                diffusion_stats_btn.click(
                    fn=self.get_diffusion_stats,
                    outputs=[diffusion_stats_output]
                )
            
            # Connect memory and history events
            memory_btn.click(
                fn=self.get_memory_statistics,
                outputs=[memory_output]
            )
            
            history_btn.click(
                fn=self.get_research_history,
                outputs=[history_output]
            )
        
        return interface
    
    def launch(self, share=False, debug=False):
        """Launch the enhanced Gradio interface"""
        interface = self.create_interface()
        
        print(f"🚀 Launching {'Diffusion-Enhanced' if self.diffusion_enabled else 'Standard'} AI Research Agent")
        print(f"🌊 Diffusion capabilities: {'✅ Enabled' if self.diffusion_enabled else '❌ Disabled'}")
        
        interface.launch(
            share=share,
            debug=debug,
            server_name="0.0.0.0",
            server_port=7861,  # Different port to avoid conflicts
            show_error=True
        )

def main():
    """Main function to run the enhanced Gradio app"""
    app = EnhancedGradioResearchInterface()
    app.launch(share=False, debug=True)

if __name__ == "__main__":
    main()