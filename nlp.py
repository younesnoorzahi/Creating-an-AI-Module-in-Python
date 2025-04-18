from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize models (lazy loading)
_sentiment_analyzer = None
_text_generator = None

def analyze_sentiment(text):
    """
    Analyze sentiment of text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment scores
    """
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        nltk.download('vader_lexicon')
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer.polarity_scores(text)

def generate_text(prompt, max_length=50):
    """
    Generate text based on prompt
    
    Args:
        prompt (str): Starting text
        max_length (int): Max length of generated text
        
    Returns:
        str: Generated text
    """
    global _text_generator
    if _text_generator is None:
        _text_generator = pipeline('text-generation', model='gpt2')
    return _text_generator(prompt, max_length=max_length)[0]['generated_text']