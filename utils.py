import numpy as np
import re

def preprocess_data(text):
    """
    Basic text preprocessing
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def normalize_features(X):
    """
    Normalize feature matrix
    
    Args:
        X (numpy.ndarray): Feature matrix
        
    Returns:
        numpy.ndarray: Normalized features
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)