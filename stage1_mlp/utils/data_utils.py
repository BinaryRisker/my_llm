"""
Data processing utilities for text classification tasks.

This module provides tools for text preprocessing, vocabulary building,
and data loading for the MLP text classification model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
import re
import json
from collections import Counter, OrderedDict
import pickle


class TextVocabulary:
    """
    Vocabulary manager for text data.
    
    Handles tokenization, vocabulary building, and text-to-index conversion.
    """
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        
        # Token to index mappings
        self.token2idx = {}
        self.idx2token = {}
        self.token_counts = Counter()
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, 
            self.START_TOKEN, self.END_TOKEN
        ]
        
        for i, token in enumerate(special_tokens):
            self.token2idx[token] = i
            self.idx2token[i] = token
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words/tokens.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Basic tokenization - can be improved with more sophisticated methods
        text = text.lower().strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts (List[str]): List of text documents
        """
        print("Building vocabulary...")
        
        # Count all tokens
        for text in texts:
            tokens = self.tokenize(text)
            self.token_counts.update(tokens)
        
        print(f"Found {len(self.token_counts)} unique tokens")
        
        # Filter by frequency and limit vocabulary size
        filtered_tokens = [
            token for token, count in self.token_counts.most_common()
            if count >= self.min_freq
        ]
        
        # Limit vocabulary size (excluding special tokens)
        max_regular_tokens = self.max_vocab_size - len(self.token2idx)
        filtered_tokens = filtered_tokens[:max_regular_tokens]
        
        # Add tokens to vocabulary
        for token in filtered_tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        
        print(f"Final vocabulary size: {len(self.token2idx)}")
    
    def text_to_indices(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Convert text to list of token indices.
        
        Args:
            text (str): Input text
            max_length (int, optional): Maximum sequence length
            
        Returns:
            List[int]: List of token indices
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to indices
        indices = []
        for token in tokens:
            idx = self.token2idx.get(token, self.token2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        # Truncate or pad to max_length if specified
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                pad_length = max_length - len(indices)
                indices.extend([self.token2idx[self.PAD_TOKEN]] * pad_length)
        
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """
        Convert list of indices back to text.
        
        Args:
            indices (List[int]): List of token indices
            
        Returns:
            str: Reconstructed text
        """
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, self.UNK_TOKEN)
            if token != self.PAD_TOKEN:
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'token_counts': dict(self.token_counts),
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.token2idx = vocab_data['token2idx']
        self.idx2token = vocab_data['idx2token']
        self.token_counts = Counter(vocab_data['token_counts'])
        self.max_vocab_size = vocab_data['max_vocab_size']
        self.min_freq = vocab_data['min_freq']
        
        print(f"Vocabulary loaded from {filepath}")
    
    def __len__(self):
        return len(self.token2idx)


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocabulary: TextVocabulary,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = self.vocabulary.text_to_indices(text, self.max_length)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [1 if i != self.vocabulary.token2idx[self.vocabulary.PAD_TOKEN] else 0 
                for i in indices]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BagOfWordsDataset(Dataset):
    """
    Dataset that converts text to bag-of-words representation.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocabulary: TextVocabulary,
        use_tfidf: bool = False
    ):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.use_tfidf = use_tfidf
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
        
        # Precompute bag-of-words vectors
        self.bow_vectors = self._compute_bow_vectors()
    
    def _compute_bow_vectors(self):
        """Compute bag-of-words vectors for all texts."""
        vocab_size = len(self.vocabulary)
        bow_vectors = []
        
        for text in self.texts:
            # Count tokens in text
            tokens = self.vocabulary.tokenize(text)
            token_counts = Counter()
            
            for token in tokens:
                idx = self.vocabulary.token2idx.get(
                    token, 
                    self.vocabulary.token2idx[self.vocabulary.UNK_TOKEN]
                )
                token_counts[idx] += 1
            
            # Create bow vector
            bow_vector = torch.zeros(vocab_size, dtype=torch.float)
            for idx, count in token_counts.items():
                bow_vector[idx] = count
            
            # Normalize if using TF-IDF (simplified version)
            if self.use_tfidf:
                bow_vector = bow_vector / (torch.sum(bow_vector) + 1e-8)
            
            bow_vectors.append(bow_vector)
        
        return bow_vectors
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_features': self.bow_vectors[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_ag_news_sample() -> Tuple[List[str], List[int], List[str]]:
    """
    Load a sample of AG News dataset.
    
    Returns:
        Tuple[List[str], List[int], List[str]]: texts, labels, class_names
    """
    # Sample AG News data (you can replace with actual dataset loading)
    sample_data = [
        ("World leaders meet to discuss climate change policies", 0),
        ("Stock market reaches new all-time high", 1),
        ("New smartphone features revolutionary camera technology", 2),
        ("Baseball season kicks off with exciting matchups", 3),
        ("Breaking news: Major earthquake hits coastal region", 0),
        ("Tech company announces breakthrough in AI research", 2),
        ("Olympic athletes prepare for upcoming games", 3),
        ("Economic indicators show strong growth trends", 1),
        ("Scientists discover new species in Amazon rainforest", 2),
        ("Political rally draws thousands of supporters", 0),
        ("Cryptocurrency market shows volatile trading patterns", 1),
        ("Football championship draws record viewership", 3),
        ("Medical researchers develop new treatment protocol", 2),
        ("International trade agreement reaches final stages", 0),
        ("Banking sector reports quarterly earnings growth", 1),
        ("Tennis tournament features top-ranked players", 3),
        ("Space mission launches with advanced technology", 2),
        ("Government announces new infrastructure spending", 0),
        ("Real estate market shows continued expansion", 1),
        ("Marathon event attracts runners from 50 countries", 3)
    ]
    
    texts = [item[0] for item in sample_data]
    labels = [item[1] for item in sample_data]
    class_names = ["World", "Business", "Sci/Tech", "Sports"]
    
    return texts, labels, class_names


def create_data_loaders(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    vocabulary: TextVocabulary,
    batch_size: int = 32,
    max_length: int = 128,
    use_bow: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        train_texts, train_labels: Training data
        test_texts, test_labels: Test data
        vocabulary: Text vocabulary
        batch_size: Batch size for data loading
        max_length: Maximum sequence length
        use_bow: Whether to use bag-of-words representation
        
    Returns:
        Tuple[DataLoader, DataLoader]: train_loader, test_loader
    """
    if use_bow:
        train_dataset = BagOfWordsDataset(train_texts, train_labels, vocabulary)
        test_dataset = BagOfWordsDataset(test_texts, test_labels, vocabulary)
    else:
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, vocabulary, max_length
        )
        test_dataset = TextClassificationDataset(
            test_texts, test_labels, vocabulary, max_length
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the data utilities
    print("🔧 Testing Data Utilities")
    print("=" * 50)
    
    # Load sample data
    texts, labels, class_names = load_ag_news_sample()
    print(f"Loaded {len(texts)} samples with {len(class_names)} classes")
    
    # Build vocabulary
    vocab = TextVocabulary(max_vocab_size=1000, min_freq=1)
    vocab.build_vocabulary(texts)
    
    # Test tokenization
    sample_text = texts[0]
    tokens = vocab.tokenize(sample_text)
    indices = vocab.text_to_indices(sample_text, max_length=20)
    reconstructed = vocab.indices_to_text(indices)
    
    print(f"\nTokenization test:")
    print(f"Original: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    print(f"Reconstructed: {reconstructed}")
    
    # Create datasets
    train_dataset = TextClassificationDataset(texts[:15], labels[:15], vocab)
    test_dataset = TextClassificationDataset(texts[15:], labels[15:], vocab)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Test data loading
    sample = train_dataset[0]
    print(f"\nSample data shapes:")
    for key, value in sample.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    
    print("\n" + "=" * 50)
    print("✅ Data utilities implementation complete!")