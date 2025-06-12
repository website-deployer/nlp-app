from flask import Flask, render_template, request, jsonify
from text_analyzer import TextAnalyzer
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for production
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

app = Flask(__name__)

def get_plot_as_base64(plt):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        analyzer = TextAnalyzer(text)
        # Get all analyses
        basic_stats = analyzer.get_basic_stats()
        sentiment = analyzer.get_sentiment_analysis()
        word_freq = analyzer.get_word_frequency(top_n=20)
        bigrams = analyzer.get_ngrams(n=2, top_n=10)
        pos_tags = analyzer.get_pos_tags()

        # Generate plots
        word_freq_plot = get_plot_as_base64(analyzer.plot_word_frequency())
        sentiment_plot = get_plot_as_base64(analyzer.plot_sentiment_distribution())

        return jsonify({
            'basic_stats': basic_stats,
            'sentiment': sentiment,
            'word_freq': word_freq,
            'bigrams': bigrams,
            'pos_tags': pos_tags,
            'word_freq_plot': word_freq_plot,
            'sentiment_plot': sentiment_plot
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 
