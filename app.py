import streamlit as st
import nltk
import ssl
import os
from collections import Counter
from textblob import TextBlob
import svgwrite
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from spellchecker import SpellChecker
import math
import random
import colorsys

# [Previous setup code remains the same...]

class AdvancedTextAnalyzer:
    def __init__(self):
        self.spell_checker = SpellChecker()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'been', 'was', 'were', 'been'}
        
        self.filler_words = {'including', 'cover', 'wide', 'range', 'topics', 'latest', 'articles', 
                            'key', 'areas', 'focus', 'include', 'importance', 'covers', 'covering'}
        
        self.domain_terms = {
            'cybersecurity': ['security', 'cyber', 'threat', 'attack', 'risk', 'vulnerability', 'breach', 'defense', 'protection'],
            'ai': ['artificial intelligence', 'machine learning', 'ai', 'ml', 'algorithm', 'automated', 'intelligent'],
            'business': ['strategy', 'management', 'program', 'implementation', 'organization', 'enterprise'],
            'technology': ['digital', 'technology', 'system', 'network', 'infrastructure', 'software', 'cloud']
        }
    
    def extract_key_concepts(self, text, num_concepts=15):
        """Extract key concepts with relationships"""
        try:
            text_lower = text.lower()
            
            # Extract important multi-word phrases
            important_phrases = []
            compound_terms = [
                'ai-driven attacks', 'ai driven attacks', 'artificial intelligence',
                'threat landscape', 'security programs', 'machine learning',
                'risk management', 'robust security', 'cyber attacks',
                'evolving threat', 'security strategies', 'data breach',
                'cybersecurity threats', 'cloud security', 'network security'
            ]
            
            found_compounds = []
            for term in compound_terms:
                if term in text_lower:
                    important_phrases.append((term.title(), 15))
                    found_compounds.append(term)
            
            # Get single words
            words = self.tokenize_words(text)
            words = [word for word in words 
                    if word.isalpha() 
                    and word not in self.stop_words 
                    and word not in self.filler_words
                    and len(word) > 2]
            
            # Boost domain-specific terms
            domain_boosted_words = []
            word_categories = {}
            
            for word in words:
                boost = 1
                for domain, terms in self.domain_terms.items():
                    if word in terms:
                        boost = 5
                        if word not in word_categories:
                            word_categories[word] = domain
                        break
                domain_boosted_words.extend([word] * boost)
            
            word_freq = Counter(domain_boosted_words)
            
            # Combine all concepts
            all_concepts = important_phrases.copy()
            
            # Filter out words that are part of compound terms
            compound_words = set()
            for compound in found_compounds:
                compound_words.update(compound.split())
            
            for word, freq in word_freq.most_common(num_concepts * 2):
                if word not in compound_words:
                    all_concepts.append((word, freq))
            
            # Sort and deduplicate
            seen = set()
            final_concepts = []
            all_concepts.sort(key=lambda x: x[1], reverse=True)
            
            for concept, score in all_concepts:
                concept_lower = concept.lower()
                if concept_lower not in seen:
                    seen.add(concept_lower)
                    # Add category information
                    category = word_categories.get(concept_lower, 'general')
                    final_concepts.append((concept, score, category))
                    if len(final_concepts) >= num_concepts:
                        break
            
            return final_concepts
            
        except Exception as e:
            return [("analysis", 1, "general"), ("text", 1, "general")]
    
    def calculate_relationships(self, text, concepts):
        """Calculate relationships between concepts"""
        sentences = self.tokenize_sentences(text)
        relationships = {}
        
        # Check which concepts appear together in sentences
        concept_words = [c[0].lower() for c in concepts]
        
        for sent in sentences:
            sent_lower = sent.lower()
            found_concepts = []
            
            for concept in concept_words:
                if concept in sent_lower:
                    found_concepts.append(concept)
            
            # Create relationships between co-occurring concepts
            for i in range(len(found_concepts)):
                for j in range(i + 1, len(found_concepts)):
                    pair = tuple(sorted([found_concepts[i], found_concepts[j]]))
                    relationships[pair] = relationships.get(pair, 0) + 1
        
        return relationships

class AdvancedSVGVisualizer:
    def __init__(self, width=1400, height=900):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Color schemes for different categories
        self.color_schemes = {
            'cybersecurity': '#e74c3c',  # Red
            'ai': '#9b59b6',              # Purple
            'business': '#3498db',         # Blue
            'technology': '#1abc9c',       # Turquoise
            'general': '#95a5a6'          # Gray
        }
        
    def get_gradient_color(self, base_color, factor):
        """Create gradient variations of a color"""
        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert to HSL and adjust lightness
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        l = min(0.9, l + (1 - l) * factor * 0.5)  # Lighten
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    def create_visualization(self, analysis_results, filename='text_analysis.svg'):
        """Create a pure visual representation"""
        dwg = svgwrite.Drawing(filename, size=(self.width, self.height))
        
        # Create gradient definitions
        defs = dwg.defs
        
        # Background gradient
        bg_gradient = dwg.linearGradient(id="bg_gradient", x1="0%", y1="0%", x2="100%", y2="100%")
        bg_gradient.add_stop_color(0, "#0f0c29")
        bg_gradient.add_stop_color(0.5, "#302b63")
        bg_gradient.add_stop_color(1, "#24243e")
        defs.add(bg_gradient)
        
        # Add animated background
        dwg.add(dwg.rect((0, 0), (self.width, self.height), fill="url(#bg_gradient)"))
        
        # Add subtle grid pattern
        for i in range(0, self.width, 50):
            dwg.add(dwg.line((i, 0), (i, self.height), stroke='#ffffff', stroke_width=0.1, opacity=0.1))
        for i in range(0, self.height, 50):
            dwg.add(dwg.line((0, i), (self.width, i), stroke='#ffffff', stroke_width=0.1, opacity=0.1))
        
        concepts = analysis_results.get('concepts', [])
        sentiment = analysis_results.get('sentiment', 0)
        relationships = analysis_results.get('relationships', {})
        
        if not concepts:
            return filename
        
        # Create central sentiment orb
        sentiment_color = '#00ff00' if sentiment > 0.1 else '#ff0000' if sentiment < -0.1 else '#ffff00'
        sentiment_radius = 80 + abs(sentiment) * 50
        
        # Sentiment core with glow effect
        for i in range(5, 0, -1):
            glow_radius = sentiment_radius + i * 10
            opacity = 0.1 * (6 - i) / 5
            dwg.add(dwg.circle(
                center=(self.center_x, self.center_y),
                r=glow_radius,
                fill=sentiment_color,
                opacity=opacity
            ))
        
        # Inner sentiment circle
        sentiment_group = dwg.g()
        sentiment_circle = dwg.circle(
            center=(self.center_x, self.center_y),
            r=sentiment_radius,
            fill=sentiment_color,
            opacity=0.3,
            stroke=sentiment_color,
            stroke_width=2
        )
        sentiment_group.add(sentiment_circle)
        
        # Add pulsing animation
        animate = dwg.animate(
            attributeName="r",
            values=f"{sentiment_radius};{sentiment_radius + 10};{sentiment_radius}",
            dur="2s",
            repeatCount="indefinite"
        )
        sentiment_circle.add(animate)
        dwg.add(sentiment_group)
        
        # Position concepts in orbital pattern
        concept_positions = []
        num_concepts = len(concepts)
        
        # Create multiple orbits based on concept importance
        orbits = 3
        concepts_per_orbit = [[], [], []]
        
        # Distribute concepts across orbits based on score
        max_score = max(c[1] for c in concepts) if concepts else 1
        
        for i, (concept, score, category) in enumerate(concepts):
            orbit_index = 0 if score > max_score * 0.7 else 1 if score > max_score * 0.3 else 2
            concepts_per_orbit[orbit_index].append((concept, score, category, i))
        
        # Draw orbital rings
        for orbit in range(orbits):
            orbit_radius = 200 + orbit * 120
            dwg.add(dwg.circle(
                center=(self.center_x, self.center_y),
                r=orbit_radius,
                fill='none',
                stroke='#ffffff',
                stroke_width=0.5,
                opacity=0.2,
                stroke_dasharray="5,5"
            ))
        
        # Place concepts on orbits
        all_positions = {}
        
        for orbit_idx, orbit_concepts in enumerate(concepts_per_orbit):
            if not orbit_concepts:
                continue
                
            orbit_radius = 200 + orbit_idx * 120
            angle_step = 2 * math.pi / len(orbit_concepts) if orbit_concepts else 0
            
            for idx, (concept, score, category, original_idx) in enumerate(orbit_concepts):
                angle = idx * angle_step
                x = self.center_x + orbit_radius * math.cos(angle)
                y = self.center_y + orbit_radius * math.sin(angle)
                
                # Store position for relationship lines
                all_positions[concept.lower()] = (x, y)
                
                # Determine size based on score
                size_factor = (score / max_score) if max_score > 0 else 0.5
                node_radius = int(30 + size_factor * 40)
                
                # Get color based on category
                base_color = self.color_schemes.get(category, self.color_schemes['general'])
                
                # Create concept node
                concept_group = dwg.g()
                
                # Outer glow
                for i in range(3, 0, -1):
                    glow_r = node_radius + i * 5
                    concept_group.add(dwg.circle(
                        center=(x, y),
                        r=glow_r,
                        fill=base_color,
                        opacity=0.1 * (4 - i) / 3
                    ))
                
                # Main node
                node = dwg.circle(
                    center=(x, y),
                    r=node_radius,
                    fill=self.get_gradient_color(base_color, 0.7),
                    stroke=base_color,
                    stroke_width=3,
                    opacity=0.8
                )
                concept_group.add(node)
                
                # Add subtle animation
                animate_scale = dwg.animateTransform(
                    attributeName="transform",
                    attributeType="XML",
                    type="scale",
                    values="1;1.1;1",
                    dur=f"{3 + orbit_idx}s",
                    repeatCount="indefinite",
                    additive="sum"
                )
                node.add(animate_scale)
                
                # Add concept text
                text_size = int(12 + size_factor * 8)
                lines = self.wrap_text(concept, node_radius * 1.8, text_size)
                y_offset = y - (len(lines) * text_size // 2)
                
                for line in lines:
                    text = dwg.text(
                        line,
                        insert=(x, y_offset),
                        text_anchor='middle',
                        style=f'font-size:{text_size}px; fill:#ffffff; font-weight:bold; font-family:Arial, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.5)'
                    )
                    concept_group.add(text)
                    y_offset += text_size + 2
                
                dwg.add(concept_group)
        
        # Draw relationships as curved lines
        if relationships:
            max_rel_strength = max(relationships.values()) if relationships else 1
            
            for (concept1, concept2), strength in relationships.items():
                if concept1 in all_positions and concept2 in all_positions:
                    x1, y1 = all_positions[concept1]
                    x2, y2 = all_positions[concept2]
                    
                    # Calculate control point for curve
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # Offset control point towards center for inward curve
                    control_x = mid_x + (self.center_x - mid_x) * 0.3
                    control_y = mid_y + (self.center_y - mid_y) * 0.3
                    
                    # Create path
                    path = dwg.path(
                        d=f"M {x1} {y1} Q {control_x} {control_y} {x2} {y2}",
                        stroke='#ffffff',
                        stroke_width=1 + (strength / max_rel_strength) * 2,
                        fill='none',
                        opacity=0.3 + (strength / max_rel_strength) * 0.4
                    )
                    
                    # Add flow animation
                    animate_dash = dwg.animate(
                        attributeName="stroke-dashoffset",
                        values="0;20",
                        dur="2s",
                        repeatCount="indefinite"
                    )
                    path.add(animate_dash)
                    path['stroke-dasharray'] = "5,5"
                    
                    dwg.add(path)
        
        # Add corner statistics visualization
        stats = analysis_results.get('statistics', {})
        if stats:
            # Create visual statistics in corner
            stat_x = 50
            stat_y = self.height - 150
            
            # Word count visualization (bar)
            word_count = stats.get('Words', 0)
            bar_width = min(200, word_count * 2)
            
            dwg.add(dwg.rect(
                (stat_x, stat_y),
                (bar_width, 20),
                fill='#3498db',
                opacity=0.7,
                rx=10
            ))
            
            # Sentence count (circles)
            sentence_count = stats.get('Sentences', 0)
            for i in range(min(sentence_count, 10)):
                dwg.add(dwg.circle(
                    center=(stat_x + 20 + i * 25, stat_y + 50),
                    r=8,
                    fill='#e74c3c',
                    opacity=0.7
                ))
        
        # Add title at top
        title = analysis_results.get('title', '')
        if title:
            title_group = dwg.g()
            
            # Title background
            title_rect = dwg.rect(
                (self.width//2 - 200, 20),
                (400, 60),
                rx=30,
                fill='#000000',
                opacity=0.5
            )
            title_group.add(title_rect)
            
            # Title text
            title_text = dwg.text(
                title,
                insert=(self.width//2, 55),
                text_anchor='middle',
                style='font-size:28px; fill:#ffffff; font-weight:bold; font-family:Arial, sans-serif; text-shadow: 3px 3px 6px rgba(0,0,0,0.7)'
            )
            title_group.add(title_text)
            
            dwg.add(title_group)
        
        # Add creation timestamp as visual signature
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        sig_group = dwg.g(opacity=0.3)
        
        # Create visual timestamp (dots pattern)
        for i, digit in enumerate(timestamp):
            digit_val = int(digit)
            for j in range(digit_val):
                sig_group.add(dwg.circle(
                    center=(self.width - 100 + i * 10, self.height - 30 - j * 3),
                    r=1,
                    fill='#ffffff'
                ))
        
        dwg.add(sig_group)
        
        dwg.save()
        return filename
    
    def wrap_text(self, text, max_width, font_size):
        """Wrap text to fit within given width"""
        words = text.split()
        lines = []
        current_line = []
        
        char_width = font_size * 0.6
        max_chars = int(max_width / char_width)
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

def analyze_text(text, title=""):
    """Main analysis function focused on visualization"""
    analyzer = AdvancedTextAnalyzer()
    visualizer = AdvancedSVGVisualizer()
    
    # Correct spelling
    corrected_text = analyzer.correct_spelling(text)
    
    # Extract concepts with categories
    concepts = analyzer.extract_key_concepts(corrected_text, num_concepts=12)
    
    # Calculate relationships
    relationships = analyzer.calculate_relationships(corrected_text, concepts)
    
    # Calculate sentiment
    sentiment, _ = analyzer.calculate_sentiment(corrected_text)
    
    # Basic statistics for visual elements
    words = analyzer.tokenize_words(corrected_text)
    sentences = analyzer.tokenize_sentences(corrected_text)
    
    # Prepare results
    analysis_results = {
        'title': title,
        'concepts': concepts,
        'relationships': relationships,
        'sentiment': sentiment,
        'statistics': {
            'Words': len(words),
            'Sentences': len(sentences)
        }
    }
    
    # Create visualization
    svg_file = visualizer.create_visualization(analysis_results)
    
    return svg_file, analysis_results

# Streamlit UI
st.set_page_config(page_title="Visual Text Analyzer", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0f0c29;
        background-image: linear-gradient(to right, #0f0c29, #302b63, #24243e);
    }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stButton button {
        background-color: #9b59b6;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #8e44ad;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Visual Text Analyzer")

# Text input
text_input = st.text_area(
    "Enter your text:",
    height=200,
    placeholder="Paste your text here for visual analysis...",
    key="text_input"
)

# Title input (optional)
title_input = st.text_input(
    "Title (optional):",
    placeholder="Give your visualization a title...",
    key="title_input"
)

# Analyze button
if st.button("🎨 Create Visualization", type="primary", use_container_width=True):
    if text_input:
        with st.spinner("Creating your visualization..."):
            try:
                # Perform analysis
                svg_file, results = analyze_text(text_input, title_input)
                
                # Display visualization
                with open(svg_file, 'r') as f:
                    svg_content = f.read()
                
                # Full-width display
                st.components.v1.html(f"""
                    <div style="display: flex; justify-content: center; align-items: center; width: 100%; background: #000;">
                        {svg_content}
                    </div>
                """, height=950)
                
                # Download button
                st.download_button(
                    label="💾 Download Visualization",
                    data=svg_content,
                    file_name=f"{title_input.replace(' ', '_') if title_input else 'visualization'}.svg",
                    mime="image/svg+xml"
                )
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
    else:
        st.warning("Please enter text to analyze!")

# Quick examples
with st.expander("📝 Load Example"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Cybersecurity"):
            st.session_state.text_input = """Cybersecurity faces unprecedented challenges with AI-driven attacks becoming more sophisticated. 
Organizations must implement robust security programs to protect against evolving threats. 
The threat landscape now includes ransomware, phishing, and advanced persistent threats. 
Cloud security and zero-trust architectures are becoming essential for modern defense strategies."""
    
    with col2:
        if st.button("AI Technology"):
            st.session_state.text_input = """Artificial intelligence and machine learning are revolutionizing industries worldwide. 
Deep learning algorithms can now process vast amounts of data with unprecedented accuracy. 
Natural language processing enables machines to understand and generate human-like text. 
Computer vision applications are transforming healthcare, autonomous vehicles, and security systems."""
    
    with col3:
        if st.button("Climate Science"):
            st.session_state.text_input = """Climate change represents an existential threat requiring immediate global action. 
Rising temperatures are causing polar ice caps to melt and sea levels to rise dramatically. 
Extreme weather events are becoming more frequent and severe across all continents. 
Renewable energy transition and carbon capture technologies offer hope for mitigation."""
