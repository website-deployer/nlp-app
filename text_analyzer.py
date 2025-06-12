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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.blob = TextBlob(text)
        self.tokens = word_tokenize(text.lower())
        self.sentences = sent_tokenize(text)
        self.stop_words = set(stopwords.words('english'))
        
    def get_basic_stats(self):
        """Get basic statistics about the text"""
        stats = {
            'Total Characters': len(self.text),
            'Total Words': len(self.tokens),
            'Total Sentences': len(self.sentences),
            'Average Word Length': sum(len(word) for word in self.tokens) / len(self.tokens),
            'Average Sentence Length': len(self.tokens) / len(self.sentences)
        }
        return stats
    
    def get_sentiment_analysis(self):
        """Analyze sentiment of the text"""
        sentiment = self.blob.sentiment
        return {
            'Polarity': sentiment.polarity,  # -1 to 1 (negative to positive)
            'Subjectivity': sentiment.subjectivity  # 0 to 1 (objective to subjective)
        }
    
    def get_word_frequency(self, top_n=10):
        """Get most common words excluding stopwords and punctuation"""
        words = [word for word in self.tokens 
                if word not in self.stop_words 
                and word not in string.punctuation]
        return dict(Counter(words).most_common(top_n))
    
    def get_ngrams(self, n=2, top_n=10):
        """Get most common n-grams"""
        n_grams = list(ngrams(self.tokens, n))
        return dict(Counter(n_grams).most_common(top_n))
    
    def get_pos_tags(self):
        """Get Part of Speech distribution"""
        pos_tags = nltk.pos_tag(self.tokens)
        pos_counts = Counter(tag for word, tag in pos_tags)
        return dict(pos_counts)
    
    def plot_word_frequency(self, top_n=10):
        """Plot word frequency distribution"""
        freq_dist = self.get_word_frequency(top_n)
        plt.figure(figsize=(12, 6))
        plt.bar(freq_dist.keys(), freq_dist.values())
        plt.title(f'Top {top_n} Most Common Words')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_distribution(self):
        """Plot sentiment distribution across sentences"""
        sentiments = [TextBlob(sentence).sentiment.polarity for sentence in self.sentences]
        plt.figure(figsize=(10, 6))
        sns.histplot(sentiments, bins=20)
        plt.title('Sentiment Distribution Across Sentences')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.show()

def analyze_text(text):
    """Main function to analyze text and return all statistics"""
    analyzer = TextAnalyzer(text)
    
    analysis_results = {
        'Basic Statistics': analyzer.get_basic_stats(),
        'Sentiment Analysis': analyzer.get_sentiment_analysis(),
        'Top 10 Words': analyzer.get_word_frequency(10),
        'Top 10 Bigrams': analyzer.get_ngrams(2, 10),
        'POS Distribution': analyzer.get_pos_tags()
    }
    
    # Generate plots
    analyzer.plot_word_frequency()
    analyzer.plot_sentiment_distribution()
    
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