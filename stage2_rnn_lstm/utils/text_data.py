"""
Text data processing utilities for RNN/LSTM training.

This module provides tools for text preprocessing, sequence generation,
and character/word-level tokenization for language modeling tasks.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
import re
import json
import pickle
import random
import numpy as np
from collections import Counter, defaultdict


class CharacterVocabulary:
    """
    Character-level vocabulary for text processing.
    
    This is useful for character-level language modeling where
    we predict the next character given previous characters.
    """
    
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.char_counts = Counter()
        
        # Special characters
        self.PAD_CHAR = '<PAD>'
        self.UNK_CHAR = '<UNK>'
        self.START_CHAR = '<START>'
        self.END_CHAR = '<END>'
        
        # Initialize with special characters
        self._init_special_chars()
    
    def _init_special_chars(self):
        """Initialize special characters in vocabulary."""
        special_chars = [self.PAD_CHAR, self.UNK_CHAR, self.START_CHAR, self.END_CHAR]
        for i, char in enumerate(special_chars):
            self.char2idx[char] = i
            self.idx2char[i] = char
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build character vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        print("Building character vocabulary...")
        
        # Count all characters
        for text in texts:
            self.char_counts.update(text)
        
        print(f"Found {len(self.char_counts)} unique characters")
        
        # Add characters to vocabulary (sorted for consistency)
        for char in sorted(self.char_counts.keys()):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
        
        print(f"Final vocabulary size: {len(self.char2idx)}")
    
    def text_to_indices(self, text: str, add_special: bool = True) -> List[int]:
        """
        Convert text to list of character indices.
        
        Args:
            text: Input text
            add_special: Whether to add START/END tokens
            
        Returns:
            List of character indices
        """
        indices = []
        
        if add_special:
            indices.append(self.char2idx[self.START_CHAR])
        
        for char in text:
            idx = self.char2idx.get(char, self.char2idx[self.UNK_CHAR])
            indices.append(idx)
        
        if add_special:
            indices.append(self.char2idx[self.END_CHAR])
        
        return indices
    
    def indices_to_text(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Convert indices back to text.
        
        Args:
            indices: List of character indices
            remove_special: Whether to remove special characters
            
        Returns:
            Reconstructed text
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, self.UNK_CHAR)
            if remove_special and char in [self.PAD_CHAR, self.START_CHAR, self.END_CHAR]:
                continue
            chars.append(char)
        
        return ''.join(chars)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'char2idx': self.char2idx,
            'idx2char': self.idx2char,
            'char_counts': dict(self.char_counts)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Character vocabulary saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = vocab_data['idx2char']
        self.char_counts = Counter(vocab_data['char_counts'])
        
        print(f"Character vocabulary loaded from {filepath}")
    
    def __len__(self):
        return len(self.char2idx)


class WordVocabulary:
    """
    Word-level vocabulary for text processing.
    
    Similar to CharacterVocabulary but works with words instead of characters.
    """
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD_WORD = '<PAD>'
        self.UNK_WORD = '<UNK>'
        self.START_WORD = '<START>'
        self.END_WORD = '<END>'
        
        # Initialize with special tokens
        self._init_special_words()
    
    def _init_special_words(self):
        """Initialize special words in vocabulary."""
        special_words = [self.PAD_WORD, self.UNK_WORD, self.START_WORD, self.END_WORD]
        for i, word in enumerate(special_words):
            self.word2idx[word] = i
            self.idx2word[i] = word
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Basic tokenization - can be improved
        text = text.lower().strip()
        text = re.sub(r'[^\\w\\s.,!?;:\\'-]', '', text)
        words = re.findall(r'\\b\\w+\\b|[.,!?;]', text)
        return words
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build word vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        print("Building word vocabulary...")
        
        # Count all words
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)
        
        print(f"Found {len(self.word_counts)} unique words")
        
        # Filter by frequency and limit vocabulary size
        filtered_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.min_freq
        ]
        
        # Limit vocabulary size (excluding special tokens)
        max_regular_words = self.max_vocab_size - len(self.word2idx)
        filtered_words = filtered_words[:max_regular_words]
        
        # Add words to vocabulary
        for word in filtered_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Final vocabulary size: {len(self.word2idx)}")
    
    def text_to_indices(self, text: str, add_special: bool = True) -> List[int]:
        """Convert text to word indices."""
        words = self.tokenize(text)
        indices = []
        
        if add_special:
            indices.append(self.word2idx[self.START_WORD])
        
        for word in words:
            idx = self.word2idx.get(word, self.word2idx[self.UNK_WORD])
            indices.append(idx)
        
        if add_special:
            indices.append(self.word2idx[self.END_WORD])
        
        return indices
    
    def indices_to_text(self, indices: List[int], remove_special: bool = True) -> str:
        """Convert indices back to text."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_WORD)
            if remove_special and word in [self.PAD_WORD, self.START_WORD, self.END_WORD]:
                continue
            words.append(word)
        
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


class LanguageModelingDataset(Dataset):
    """
    Dataset for language modeling tasks.
    
    This creates input-target pairs where the target is the input
    shifted by one position (next token prediction).
    """
    
    def __init__(
        self,
        texts: List[str],
        vocabulary: Union[CharacterVocabulary, WordVocabulary],
        seq_length: int = 64,
        stride: int = 32
    ):
        self.texts = texts
        self.vocabulary = vocabulary
        self.seq_length = seq_length
        self.stride = stride
        
        # Convert all texts to indices
        self.sequences = self._prepare_sequences()
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def _prepare_sequences(self) -> List[List[int]]:
        """Prepare sequences for training."""
        sequences = []
        
        for text in self.texts:
            # Convert text to indices
            indices = self.vocabulary.text_to_indices(text, add_special=False)
            
            # Create overlapping sequences
            for i in range(0, len(indices) - self.seq_length, self.stride):
                seq = indices[i:i + self.seq_length + 1]  # +1 for target
                if len(seq) == self.seq_length + 1:
                    sequences.append(seq)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Input: all tokens except the last
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        # Target: all tokens except the first (shifted by 1)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input': input_seq,
            'target': target_seq
        }


class TextGenerationDataset(Dataset):
    """
    Dataset for conditional text generation.
    
    This pairs input prompts with target completions.
    """
    
    def __init__(
        self,
        prompts: List[str],
        completions: List[str],
        vocabulary: Union[CharacterVocabulary, WordVocabulary],
        max_input_length: int = 64,
        max_target_length: int = 128
    ):
        self.prompts = prompts
        self.completions = completions
        self.vocabulary = vocabulary
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        assert len(prompts) == len(completions), "Prompts and completions must have same length"
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        completion = self.completions[idx]
        
        # Convert to indices
        input_indices = self.vocabulary.text_to_indices(prompt, add_special=True)
        target_indices = self.vocabulary.text_to_indices(completion, add_special=True)
        
        # Truncate if too long
        input_indices = input_indices[:self.max_input_length]
        target_indices = target_indices[:self.max_target_length]
        
        # Pad to fixed lengths
        pad_idx = self.vocabulary.char2idx.get(self.vocabulary.PAD_CHAR, 0) if hasattr(self.vocabulary, 'char2idx') else 0
        
        input_padded = input_indices + [pad_idx] * (self.max_input_length - len(input_indices))
        target_padded = target_indices + [pad_idx] * (self.max_target_length - len(target_indices))
        
        return {
            'input': torch.tensor(input_padded, dtype=torch.long),
            'target': torch.tensor(target_padded, dtype=torch.long),
            'input_length': len(input_indices),
            'target_length': len(target_indices)
        }


def load_text_file(filepath: str) -> str:
    """Load text from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()


def create_sample_text() -> List[str]:
    """
    Create sample text data for demonstration.
    
    Returns:
        List of sample texts
    """
    sample_texts = [
        # Classical literature style
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.",
        
        # Scientific text
        "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. It consists of two interrelated theories: special relativity and general relativity.",
        
        # News style
        "Scientists at the Large Hadron Collider have made a groundbreaking discovery that could change our understanding of particle physics. The experiment involved colliding particles at unprecedented energies.",
        
        # Conversational text
        "Hey there! How are you doing today? I was just thinking about our conversation yesterday and I wanted to follow up on that idea we discussed about machine learning.",
        
        # Technical documentation
        "To initialize the neural network, first create an instance of the model class. Then, define the loss function and optimizer. Finally, iterate through the training data and update the model parameters.",
        
        # Creative writing
        "The moon hung low in the sky, casting silver shadows across the ancient forest. Sarah could hear the whisper of wind through the leaves and the distant call of an owl.",
        
        # Historical text
        "The Renaissance period, spanning roughly from the 14th to the 17th century, marked a time of great cultural, artistic, and intellectual achievement in European history.",
        
        # Poetry
        "Roses are red, violets are blue, sugar is sweet, and so are you. The stars shine bright in the midnight sky, while gentle breezes whisper by."
    ]
    
    return sample_texts


def split_text_data(
    texts: List[str], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split text data into train/validation/test sets.
    
    Args:
        texts: List of texts
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_texts, val_texts, test_texts)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Shuffle texts
    texts = texts.copy()
    random.shuffle(texts)
    
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    return train_texts, val_texts, test_texts


def create_data_loaders(
    train_texts: List[str],
    val_texts: List[str],
    vocabulary: Union[CharacterVocabulary, WordVocabulary],
    seq_length: int = 64,
    batch_size: int = 32,
    stride: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        vocabulary: Vocabulary object
        seq_length: Sequence length
        batch_size: Batch size
        stride: Stride for sequence creation (default: seq_length // 2)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if stride is None:
        stride = seq_length // 2
    
    # Create datasets
    train_dataset = LanguageModelingDataset(
        train_texts, vocabulary, seq_length, stride
    )
    val_dataset = LanguageModelingDataset(
        val_texts, vocabulary, seq_length, stride
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()


if __name__ == "__main__":
    # Test the data processing utilities
    print("📚 Testing Text Data Processing")
    print("=" * 50)
    
    # Create sample data
    texts = create_sample_text()
    print(f"Created {len(texts)} sample texts")
    
    # Test character vocabulary
    print("\\nTesting Character Vocabulary:")
    char_vocab = CharacterVocabulary()
    char_vocab.build_vocabulary(texts)
    
    sample_text = texts[0][:50]  # First 50 characters
    char_indices = char_vocab.text_to_indices(sample_text)
    reconstructed = char_vocab.indices_to_text(char_indices)
    
    print(f"Original: {sample_text}")
    print(f"Indices: {char_indices}")
    print(f"Reconstructed: {reconstructed}")
    
    # Test word vocabulary
    print("\\nTesting Word Vocabulary:")
    word_vocab = WordVocabulary(max_vocab_size=1000, min_freq=1)
    word_vocab.build_vocabulary(texts)
    
    word_indices = word_vocab.text_to_indices(texts[0])
    word_reconstructed = word_vocab.indices_to_text(word_indices)
    print(f"Word tokenization: {word_vocab.tokenize(texts[0][:100])}")
    print(f"Word reconstruction: {word_reconstructed[:100]}")
    
    # Test dataset creation
    print("\\nTesting Dataset Creation:")
    train_texts, val_texts, test_texts = split_text_data(texts)
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Create language modeling dataset
    dataset = LanguageModelingDataset(train_texts, char_vocab, seq_length=32)
    sample = dataset[0]
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample input shape: {sample['input'].shape}")
    print(f"Sample target shape: {sample['target'].shape}")
    print(f"Sample input text: {char_vocab.indices_to_text(sample['input'].tolist())}")
    print(f"Sample target text: {char_vocab.indices_to_text(sample['target'].tolist())}")
    
    # Test data loaders
    train_loader, val_loader = create_data_loaders(
        train_texts, val_texts, char_vocab, seq_length=32, batch_size=4
    )
    
    print(f"\\nData loaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch input shape: {batch['input'].shape}")
        print(f"Batch target shape: {batch['target'].shape}")
        break
    
    print("\\n✅ Text data processing tests complete!")