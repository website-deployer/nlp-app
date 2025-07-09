import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import ngrams
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class TextAnalyzer:
    def __init__(self, text):
        self.original_text = text
        self.cleaned_text = self._clean_text(text)
        self.blob = TextBlob(self.cleaned_text)
        self.tokens = word_tokenize(self.cleaned_text.lower())
        self.sentences = sent_tokenize(self.cleaned_text)
        self.stop_words = set(stopwords.words('english'))
        
    def _clean_text(self, text):
        """Clean the input text by removing unwanted characters and normalizing whitespace"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove possessives
        text = re.sub(r"'s\b", '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def get_basic_stats(self):
        """Get basic statistics about the text"""
        if not self.tokens or not self.sentences:
            return {
                'Total Characters': len(self.original_text),
                'Total Words': 0,
                'Total Sentences': 0,
                'Average Word Length': 0.0,
                'Average Sentence Length': 0.0,
                'Unique Words': 0
            }
            
        # Get meaningful words (excluding stopwords and single characters)
        meaningful_words = [word for word in self.tokens 
                          if word not in self.stop_words 
                          and len(word) > 1]
        
        stats = {
            'Total Characters': len(self.original_text),
            'Total Words': len(meaningful_words),
            'Total Sentences': len(self.sentences),
            'Average Word Length': sum(len(word) for word in meaningful_words) / len(meaningful_words) if meaningful_words else 0.0,
            'Average Sentence Length': len(meaningful_words) / len(self.sentences) if self.sentences else 0.0,
            'Unique Words': len(set(meaningful_words))
        }
        return stats
    
    def get_sentiment_analysis(self):
        """Analyze sentiment of the text with improved accuracy"""
        # Analyze sentiment for each sentence
        sentence_sentiments = [TextBlob(sentence).sentiment for sentence in self.sentences]
        
        # Calculate weighted average based on sentence length
        total_length = sum(len(sentence) for sentence in self.sentences)
        weighted_polarity = sum(getattr(sent, 'polarity', 0) * len(sentence) for sent, sentence in zip(sentence_sentiments, self.sentences)) / total_length if total_length > 0 else 0
        weighted_subjectivity = sum(getattr(sent, 'subjectivity', 0) * len(sentence) for sent, sentence in zip(sentence_sentiments, self.sentences)) / total_length if total_length > 0 else 0
        
        return {
            'Polarity': round(weighted_polarity, 3),
            'Subjectivity': round(weighted_subjectivity, 3),
            'Sentiment': 'Positive' if weighted_polarity > 0.1 else 'Negative' if weighted_polarity < -0.1 else 'Neutral'
        }
    
    def get_word_frequency(self, top_n=10):
        """Get most common words with improved filtering"""
        # Filter out stopwords, single characters, and numbers
        words = [word for word in self.tokens 
                if word not in self.stop_words 
                and len(word) > 1
                and not word.isdigit()]
        
        # Get frequency distribution
        freq_dist = Counter(words)
        
        # Return top N words with their frequencies and percentages
        total_words = sum(freq_dist.values())
        if total_words == 0:
            return {}
        return {
            word: {
                'count': count,
                'percentage': round((count / total_words) * 100, 1)
            }
            for word, count in freq_dist.most_common(top_n)
        }
    
    def get_ngrams(self, n=2, top_n=10):
        """Get most common n-grams with improved filtering"""
        # Filter tokens before creating n-grams
        filtered_tokens = [word for word in self.tokens 
                         if word not in self.stop_words 
                         and len(word) > 1
                         and not word.isdigit()]
        
        n_grams = list(ngrams(filtered_tokens, n))
        freq_dist = Counter(n_grams)
        
        # Return top N n-grams with their frequencies and percentages
        total_ngrams = sum(freq_dist.values())
        if total_ngrams == 0:
            return {}
        return {
            ' '.join(gram): {
                'count': count,
                'percentage': round((count / total_ngrams) * 100, 1)
            }
            for gram, count in freq_dist.most_common(top_n)
        }
    
    def get_pos_tags(self):
        """Get Part of Speech distribution"""
        pos_tags = nltk.pos_tag(self.tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        return dict(pos_counts)
    
    def plot_word_frequency(self, top_n=10):
        """Plot word frequency distribution with improved visualization"""
        freq_dist = self.get_word_frequency(top_n)
        words = list(freq_dist.keys())
        counts = [data['count'] for data in freq_dist.values()]
        percentages = [data['percentage'] for data in freq_dist.values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(words, counts)
        
        # Add percentage labels on top of bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage}%',
                   ha='center', va='bottom')
        
        ax.set_title(f'Top {top_n} Most Common Words')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_sentiment_distribution(self):
        """Plot sentiment distribution with improved visualization"""
        sentiments = [float(TextBlob(sentence).sentiment.polarity) for sentence in self.sentences]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if sentiments:
            sns.histplot(sentiments, bins=20, ax=ax, color='#007AFF')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add mean line
            mean_sentiment = sum(sentiments) / len(sentiments)
            ax.axvline(x=mean_sentiment, color='red', linestyle='-', alpha=0.5,
                      label=f'Mean: {mean_sentiment:.2f}')
            
            ax.legend()
        
        ax.set_title('Sentiment Distribution Across Sentences')
        ax.set_xlabel('Sentiment Polarity')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        return fig
    
    def get_readability_metrics(self):
        """Calculate various readability metrics for the text"""
        # Get meaningful words (excluding stopwords and single characters)
        meaningful_words = [word for word in self.tokens 
                          if word not in self.stop_words 
                          and len(word) > 1]
        
        if not meaningful_words or not self.sentences:
            return {
                'Flesch Reading Ease': 0,
                'Flesch-Kincaid Grade': 0,
                'Lexical Diversity': 0,
                'Complex Words': 0
            }
            
        # Calculate average sentence length using meaningful words
        avg_sentence_length = len(meaningful_words) / len(self.sentences)
        
        # Calculate average syllables per word
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            previous_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_is_vowel:
                    count += 1
                previous_is_vowel = is_vowel
            if word.endswith("e"):
                count -= 1
            return max(1, count)
            
        total_syllables = sum(count_syllables(word) for word in meaningful_words)
        avg_syllables = total_syllables / len(meaningful_words)
        
        # Calculate Flesch Reading Ease Score (0-100, higher is easier to read)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0 and 100
        
        # Calculate Flesch-Kincaid Grade Level (0-12, represents US grade level)
        fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
        fk_grade = max(0, min(12, fk_grade))  # Clamp between 0 and 12
        
        # Calculate Lexical Diversity (Type-Token Ratio)
        unique_words = len(set(meaningful_words))
        total_words = len(meaningful_words)
        lexical_diversity = (unique_words / total_words) * 100 if total_words > 0 else 0
        
        # Count complex words (words with more than 2 syllables)
        complex_words = sum(1 for word in meaningful_words if count_syllables(word) > 2)
        complex_word_percentage = (complex_words / total_words) * 100 if total_words > 0 else 0
        
        return {
            'Flesch Reading Ease': round(flesch_score, 1),
            'Flesch-Kincaid Grade': round(fk_grade, 1),
            'Lexical Diversity': round(lexical_diversity, 1),
            'Complex Words': round(complex_word_percentage, 1)
        }

def analyze_text(text):
    """Main function to analyze text and return all statistics"""
    analyzer = TextAnalyzer(text)
    
    analysis_results = {
        'basic_stats': analyzer.get_basic_stats(),
        'sentiment': analyzer.get_sentiment_analysis(),
        'word_freq': analyzer.get_word_frequency(10),
        'bigrams': analyzer.get_ngrams(2, 10),
        'readability': analyzer.get_readability_metrics()
    }
    
    # Generate plots
    word_freq_fig = analyzer.plot_word_frequency()
    sentiment_fig = analyzer.plot_sentiment_distribution()
    
    # Convert plots to base64
    import io
    import base64
    
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    analysis_results['word_freq_plot'] = fig_to_base64(word_freq_fig)
    analysis_results['sentiment_plot'] = fig_to_base64(sentiment_fig)
    
    return analysis_results

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Natural Language Processing (NLP) is a fascinating field of artificial intelligence 
    that focuses on the interaction between computers and human language. It enables 
    machines to read, understand, and derive meaning from human languages. NLP combines 
    computational linguistics with statistical, machine learning, and deep learning models.
    """
    
    results = analyze_text(sample_text)
    
    # Print results in a formatted way
    for category, stats in results.items():
        print(f"\n{category}:")
        print("-" * len(category))
        for key, value in stats.items():
            print(f"{key}: {value}") 
