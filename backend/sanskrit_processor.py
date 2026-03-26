"""
Sanskrit Text Processing Module
Handles Sanskrit text tokenization, normalization, and preprocessing
"""
import re
import nltk
from typing import List, Tuple
import numpy as np

# Download required NLTK datasets
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SanskritProcessor:
    """Process Sanskrit text for NLP tasks"""
    
    def __init__(self):
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\u0900-\u097F\s]')
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize Sanskrit text
        - Convert to lowercase
        - Remove extra spaces
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize Sanskrit text into words"""
        tokens = nltk.word_tokenize(text)
        return [token for token in tokens if token.isalpha()]
    
    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        """Remove common Sanskrit stopwords"""
        # Common Sanskrit stopwords
        stopwords = {
            'iti', 'api', 'ca', 'yatha', 'atra', 'asya', 'tasya',
            'hi', 'tu', 'vā', 'kila', 'khalu', 'eva', 'cha'
        }
        return [token for token in tokens if token.lower() not in stopwords]
    
    def preprocess(self, text: str) -> List[str]:
        """Complete preprocessing pipeline"""
        # Normalize
        text = self.normalize_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        return tokens
    
    @staticmethod
    def detect_devanagari(text: str) -> bool:
        """Detect if text contains Devanagari script"""
        devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
        return len(devanagari_chars) > 0
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Sanskrit sentences often end with danda (।) or double danda (॥)
        text = text.replace('।', '. ')
        text = text.replace('॥', '. ')
        
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]


class SanskritSummarizer:
    """Summarize Sanskrit text"""
    
    def __init__(self, processor: SanskritProcessor = None):
        self.processor = processor or SanskritProcessor()
    
    @staticmethod
    def extract_key_sentences(text: str, num_sentences: int = 3) -> List[str]:
        """
        Extract key sentences based on word frequency
        """
        processor = SanskritProcessor()
        sentences = processor.split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Calculate word frequencies
        word_freq = {}
        for sentence in sentences:
            tokens = processor.tokenize(sentence.lower())
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Score sentences
        sentence_scores = []
        for sentence in sentences:
            score = 0
            tokens = processor.tokenize(sentence.lower())
            for token in tokens:
                score += word_freq.get(token, 0)
            sentence_scores.append((sentence, score))
        
        # Get top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
        
        return [s[0] for s in top_sentences]
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate summary of Sanskrit text"""
        key_sentences = self.extract_key_sentences(text, num_sentences)
        return ' '.join(key_sentences)
