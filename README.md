# Text Analysis Application

This is a Natural Language Processing (NLP) application that provides comprehensive analysis of text input, including various statistics and visualizations. The application is available both as a Python module and as a web application.

## Features

- Basic text statistics (character count, word count, sentence count, etc.)
- Sentiment analysis (polarity and subjectivity)
- Word frequency analysis
- N-gram analysis
- Part of Speech (POS) tagging
- Visual representations of word frequency and sentiment distribution
- Modern web interface for easy text analysis

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application

1. Start the web server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter your text in the text area and click "Analyze Text" to see the results.

### Python Module

You can also use the text analyzer in your Python code:
```python
from text_analyzer import analyze_text

text = "Your text to analyze here"
results = analyze_text(text)
```

## Output

The application provides:
- Detailed statistics about the text
- Sentiment analysis results
- Word frequency distribution
- N-gram analysis
- Part of Speech distribution
- Visual plots for word frequency and sentiment distribution

## Dependencies

- nltk
- textblob
- pandas
- matplotlib
- seaborn
- flask 