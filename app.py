import streamlit as st
import nltk
import ssl
import os

# Fix SSL certificate issue and download NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data at startup
@st.cache_resource
def download_nltk_data():
    """Download all required NLTK data"""
    try:
        # Create nltk_data directory if it doesn't exist
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Download required packages
        packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'brown', 'wordnet']
        for package in packages:
            try:
                nltk.download(package, quiet=True)
            except:
                pass
        
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Initialize NLTK downloads
download_nltk_data()

# Now import the rest
from collections import Counter
from textblob import TextBlob
import svgwrite
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from spellchecker import SpellChecker
import math
import base64

# Import NLTK modules after download
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    st.warning("NLTK modules not fully loaded. Some features may be limited.")

class AdvancedTextAnalyzer:
    def __init__(self):
        self.spell_checker = SpellChecker()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK fails
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'been', 'was', 'were', 'been'}
        
    def correct_spelling(self, text):
        """Correct spelling errors in the text"""
        try:
            blob = TextBlob(text)
            corrected = str(blob.correct())
            return corrected
        except:
            return text
    
    def tokenize_words(self, text):
        """Tokenize words with fallback"""
        try:
            return word_tokenize(text.lower())
        except:
            # Simple fallback tokenization
            import re
            return re.findall(r'\b\w+\b', text.lower())
    
    def tokenize_sentences(self, text):
        """Tokenize sentences with fallback"""
        try:
            return sent_tokenize(text)
        except:
            # Simple fallback sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def extract_key_concepts(self, text, num_concepts=10):
        """Extract key concepts using TF-IDF and other metrics"""
        try:
            # Tokenize and clean
            words = self.tokenize_words(text)
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            
            if not words:
                return []
            
            # Get word frequencies
            word_freq = Counter(words)
            
            # Extract noun phrases
            try:
                blob = TextBlob(text)
                noun_phrases = [str(phrase) for phrase in blob.noun_phrases]
            except:
                noun_phrases = []
            
            # Calculate TF-IDF for sentences
            sentences = self.tokenize_sentences(text)
            tfidf_keywords = []
            
            if len(sentences) > 1:
                try:
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_concepts)
                    tfidf_matrix = vectorizer.fit_transform(sentences)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get average TF-IDF scores
                    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                    top_indices = avg_scores.argsort()[-num_concepts:][::-1]
                    tfidf_keywords = [feature_names[i] for i in top_indices]
                except:
                    pass
            
            # Combine different metrics
            all_keywords = list(word_freq.most_common(num_concepts))
            if noun_phrases:
                all_keywords.extend([(phrase, 1) for phrase in noun_phrases[:5]])
            if tfidf_keywords:
                all_keywords.extend([(word, 1) for word in tfidf_keywords])
            
            # Remove duplicates and return top concepts
            seen = set()
            final_concepts = []
            for word, score in all_keywords:
                if word not in seen and len(word) > 2:
                    seen.add(word)
                    final_concepts.append((word, score))
                    if len(final_concepts) >= num_concepts:
                        break
            
            return final_concepts if final_concepts else [("analysis", 1), ("text", 1)]
            
        except Exception as e:
            st.error(f"Error in concept extraction: {e}")
            # Return some default concepts
            return [("analysis", 1), ("text", 1)]
    
    def calculate_sentiment(self, text):
        """Calculate sentiment polarity and subjectivity"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0, 0
    
    def extract_summary_points(self, text, num_points=3):
        """Extract key summary points from the text"""
        try:
            sentences = self.tokenize_sentences(text)
            if not sentences:
                return ["No summary available."]
                
            if len(sentences) <= num_points:
                return sentences
            
            # Score sentences based on word frequency
            words = self.tokenize_words(text)
            word_freq = Counter(words)
            sentence_scores = {}
            
            for sentence in sentences:
                sentence_words = self.tokenize_words(sentence)
                if sentence_words:
                    score = sum(word_freq.get(word, 0) for word in sentence_words)
                    sentence_scores[sentence] = score / len(sentence_words)
                else:
                    sentence_scores[sentence] = 0
            
            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            return [sent[0] for sent in top_sentences[:num_points]]
        except:
            return [text[:100] + "..."] if len(text) > 100 else [text]

class SVGVisualizer:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.used_positions = []
        
    def check_overlap(self, x, y, width, height, padding=10):
        """Check if a rectangle overlaps with existing elements"""
        for pos in self.used_positions:
            px, py, pw, ph = pos
            if not (x + width + padding < px or x > px + pw + padding or
                    y + height + padding < py or y > py + ph + padding):
                return True
        return False
    
    def find_non_overlapping_position(self, width, height, center_x=None, center_y=None):
        """Find a position that doesn't overlap with existing elements"""
        if center_x is None:
            center_x = self.width // 2
        if center_y is None:
            center_y = self.height // 2
            
        # Try positions in a spiral pattern
        angle = 0
        radius = 0
        while radius < max(self.width, self.height):
            x = center_x + radius * math.cos(angle) - width // 2
            y = center_y + radius * math.sin(angle) - height // 2
            
            # Keep within bounds
            x = max(20, min(x, self.width - width - 20))
            y = max(20, min(y, self.height - height - 20))
            
            if not self.check_overlap(x, y, width, height):
                self.used_positions.append((x, y, width, height))
                return x, y
                
            angle += 0.3
            radius += 5
            
        return 50, 50  # Default position
    
    def wrap_text(self, text, max_width, font_size):
        """Wrap text to fit within a given width"""
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
    
    def create_visualization(self, analysis_results, filename='text_analysis.svg'):
        """Create an SVG visualization of the analysis results"""
        dwg = svgwrite.Drawing(filename, size=(self.width, self.height))
        
        # Add background
        dwg.add(dwg.rect((0, 0), (self.width, self.height), fill='#f8f9fa'))
        
        # Add title
        title = analysis_results.get('title', 'Text Analysis Visualization')
        dwg.add(dwg.text(title, insert=(self.width//2, 40), 
                         text_anchor='middle', 
                         style='font-size:28px; font-weight:bold; fill:#2c3e50; font-family:Arial'))
        
        # Add sentiment indicator
        sentiment = analysis_results.get('sentiment', 0)
        sentiment_color = '#27ae60' if sentiment > 0.1 else '#e74c3c' if sentiment < -0.1 else '#95a5a6'
        sentiment_text = 'Positive' if sentiment > 0.1 else 'Negative' if sentiment < -0.1 else 'Neutral'
        
        sentiment_group = dwg.g()
        sentiment_rect = dwg.rect((50, 80), (200, 60), rx=5, ry=5, fill=sentiment_color, opacity=0.2)
        sentiment_group.add(sentiment_rect)
        sentiment_group.add(dwg.text(f'Sentiment: {sentiment_text}', 
                                    insert=(150, 115), 
                                    text_anchor='middle',
                                    style='font-size:16px; fill:#2c3e50; font-family:Arial'))
        dwg.add(sentiment_group)
        self.used_positions.append((50, 80, 200, 60))
        
        # Add key concepts
        concepts = analysis_results.get('concepts', [])
        if concepts:
            max_score = max(score for _, score in concepts) if concepts else 1
            
            for i, (concept, score) in enumerate(concepts[:8]):
                size_factor = (score / max_score) if max_score > 0 else 0.5
                bubble_radius = int(40 + size_factor * 60)
                text_size = int(14 + size_factor * 12)
                
                x, y = self.find_non_overlapping_position(
                    bubble_radius * 2, bubble_radius * 2,
                    self.width // 2, self.height // 2
                )
                
                concept_group = dwg.g()
                concept_group.add(dwg.circle(
                    center=(x + bubble_radius, y + bubble_radius),
                    r=bubble_radius,
                    fill='#3498db',
                    opacity=0.3 + size_factor * 0.4
                ))
                
                lines = self.wrap_text(concept, bubble_radius * 1.5, text_size)
                y_offset = y + bubble_radius - (len(lines) * text_size // 2)
                
                for line in lines:
                    concept_group.add(dwg.text(
                        line,
                        insert=(x + bubble_radius, y_offset),
                        text_anchor='middle',
                        style=f'font-size:{text_size}px; fill:#2c3e50; font-weight:bold; font-family:Arial'
                    ))
                    y_offset += text_size + 2
                
                dwg.add(concept_group)
        
        # Add key takeaways
        takeaways = analysis_results.get('summary', [])
        if takeaways:
            takeaway_y = self.height - 250
            takeaway_group = dwg.g()
            
            takeaway_rect = dwg.rect((50, takeaway_y), (self.width - 100, 200), 
                                   rx=10, ry=10, fill='#ecf0f1', opacity=0.8)
            takeaway_group.add(takeaway_rect)
            
            takeaway_group.add(dwg.text('Key Takeaways:', 
                                       insert=(self.width//2, takeaway_y + 30),
                                       text_anchor='middle',
                                       style='font-size:20px; font-weight:bold; fill:#2c3e50; font-family:Arial'))
            
            y_pos = takeaway_y + 60
            for i, takeaway in enumerate(takeaways[:3]):
                wrapped_lines = self.wrap_text(takeaway, self.width - 200, 14)
                for line in wrapped_lines:
                    takeaway_group.add(dwg.text(
                        f"• {line}" if wrapped_lines.index(line) == 0 else f"  {line}",
                        insert=(80, y_pos),
                        style='font-size:14px; fill:#34495e; font-family:Arial'
                    ))
                    y_pos += 20
                y_pos += 10
            
            dwg.add(takeaway_group)
        
        # Add statistics
        stats = analysis_results.get('statistics', {})
        if stats:
            stats_group = dwg.g()
            stats_x = self.width - 250
            stats_y = 80
            
            stats_rect = dwg.rect((stats_x, stats_y), (200, 120), 
                                rx=5, ry=5, fill='#e8f5e9', opacity=0.8)
            stats_group.add(stats_rect)
            
            stats_group.add(dwg.text('Statistics', 
                                   insert=(stats_x + 100, stats_y + 25),
                                   text_anchor='middle',
                                   style='font-size:16px; font-weight:bold; fill:#2c3e50; font-family:Arial'))
            
            y_offset = stats_y + 50
            for key, value in stats.items():
                stats_group.add(dwg.text(f'{key}: {value}', 
                                       insert=(stats_x + 10, y_offset),
                                       style='font-size:12px; fill:#424242; font-family:Arial'))
                y_offset += 20
            
            dwg.add(stats_group)
        
        dwg.save()
        return filename

def analyze_text(text, title="Text Analysis"):
    """Main function to analyze text and create visualization"""
    analyzer = AdvancedTextAnalyzer()
    visualizer = SVGVisualizer()
    
    # Correct spelling
    corrected_text = analyzer.correct_spelling(text)
    
    # Perform analysis
    concepts = analyzer.extract_key_concepts(corrected_text, num_concepts=10)
    sentiment, subjectivity = analyzer.calculate_sentiment(corrected_text)
    summary = analyzer.extract_summary_points(corrected_text, num_points=3)
    
    # Calculate statistics
    words = analyzer.tokenize_words(corrected_text)
    sentences = analyzer.tokenize_sentences(corrected_text)
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Prepare results
    analysis_results = {
        'title': title,
        'concepts': concepts,
        'sentiment': sentiment,
        'summary': summary,
        'statistics': {
            'Words': word_count,
            'Sentences': sentence_count,
            'Avg Word Length': f'{avg_word_length:.1f}'
        }
    }
    
    # Create visualization
    svg_file = visualizer.create_visualization(analysis_results)
    
    return svg_file, analysis_results

# Streamlit UI
st.set_page_config(page_title="Text Analyzer & Visualizer", layout="wide")

st.title("🔍 Advanced Text Analyzer & Visualizer")
st.markdown("Analyze your text and get beautiful SVG visualizations with key insights!")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    # Text input
    text_input = st.text_area(
        "Enter your text here:",
        height=300,
        placeholder="Paste or type your text here...",
        help="Enter any text you want to analyze"
    )
    
    # Title input
    title_input = st.text_input(
        "Title for your analysis:",
        value="Text Analysis",
        help="This will appear at the top of your visualization"
    )
    
    # Analyze button
    analyze_button = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

with col2:
    # Sample texts
    st.subheader("📝 Sample Texts")
    
    if st.button("Load AI Example"):
        text_input = """Artificial intelligence is rapidly transforming how we live and work. 
Machine learning algorithms can now recognize patterns in vast amounts of data, 
enabling breakthroughs in healthcare, finance, and transportation. 
However, we must carefully consider the ethical implications of AI systems 
and ensure they are developed responsibly. The future of AI holds immense 
promise, but also requires thoughtful governance and human oversight."""
    
    if st.button("Load Climate Example"):
        text_input = """Climate change represents one of the most pressing challenges of our time. 
Rising global temperatures are causing sea levels to rise, weather patterns to shift, 
and ecosystems to be disrupted. Immediate action is required to reduce greenhouse gas emissions 
and transition to renewable energy sources. The next decade is critical for implementing 
sustainable solutions and protecting our planet for future generations."""
    
    if st.button("Load Business Example"):
        text_input = """The modern business landscape demands agility and innovation. 
Companies must adapt quickly to changing market conditions and customer expectations. 
Digital transformation is no longer optional but essential for survival. 
Organizations that embrace data-driven decision making and invest in their employees' 
skills will be best positioned for long-term success."""

# Analysis section
if analyze_button and text_input:
    with st.spinner("🔄 Analyzing your text..."):
        try:
            # Perform analysis
            svg_file, results = analyze_text(text_input, title_input)
            
            st.success("✅ Analysis complete!")
            
            # Display visualization
            st.subheader("📊 Visualization")
            
            # Read and display SVG
            with open(svg_file, 'r') as f:
                svg_content = f.read()
            
            # Display SVG using HTML
            st.components.v1.html(f"""
                <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                    {svg_content}
                </div>
            """, height=850)
            
            # Download button
            st.download_button(
                label="⬇️ Download SVG",
                data=svg_content,
                file_name=f"{title_input.replace(' ', '_')}_analysis.svg",
                mime="image/svg+xml"
            )
            
            # Show analysis details
            st.subheader("📈 Analysis Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = results['sentiment']
                sentiment_label = 'Positive 😊' if sentiment > 0.1 else 'Negative 😞' if sentiment < -0.1 else 'Neutral 😐'
                st.metric("Sentiment", sentiment_label, f"{sentiment:.3f}")
            
            with col2:
                st.metric("Word Count", results['statistics']['Words'])
            
            with col3:
                st.metric("Sentences", results['statistics']['Sentences'])
            
            # Top concepts
            st.subheader("🎯 Top Concepts")
            if results['concepts']:
                concepts_text = ", ".join([f"**{concept}** ({score:.0f})" for concept, score in results['concepts'][:5]])
                st.markdown(concepts_text)
            else:
                st.info("No key concepts extracted. Try a longer text.")
            
            # Key takeaways
            st.subheader("💡 Key Takeaways")
            for i, takeaway in enumerate(results['summary'], 1):
                st.write(f"{i}. {takeaway}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please make sure your text is in English and try again.")

elif analyze_button:
    st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Free Text Analysis Tool")
