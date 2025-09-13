#!/usr/bin/env python3
"""
Stage 3: Attention Mechanism and Seq2Seq - Visualization Tools
================================================================

This script provides comprehensive visualization tools for analyzing attention mechanisms
and sequence-to-sequence models, including:
- Attention heatmaps visualization
- Translation examples with alignment
- BLEU score comparisons
- Model performance analysis
- Interactive translation demo

Author: AI Assistant
Date: 2024
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Seq2SeqModel
from utils.data_loader import TranslationDataset, create_vocab
from utils.bleu_eval import compute_bleu
from utils.training import translate_sentence

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AttentionVisualizer:
    """Comprehensive attention visualization toolkit"""
    
    def __init__(self, model_path: str, vocab_path: str = None):
        """Initialize the visualizer with trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and vocabulary
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = checkpoint['model']
        self.model.eval()
        
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.src_vocab = vocab_data['src_vocab']
                self.tgt_vocab = vocab_data['tgt_vocab']
        else:
            # Create default vocabulary if not provided
            print("Warning: No vocabulary file found. Creating default vocabulary.")
            self.src_vocab, self.tgt_vocab = self._create_default_vocab()
        
        self.src_word2id = {word: i for i, word in enumerate(self.src_vocab)}
        self.tgt_word2id = {word: i for i, word in enumerate(self.tgt_vocab)}
        self.src_id2word = {i: word for word, i in self.src_word2id.items()}
        self.tgt_id2word = {i: word for word, i in self.tgt_word2id.items()}
        
        print(f"✅ Loaded model from {model_path}")
        print(f"📊 Source vocab size: {len(self.src_vocab)}")
        print(f"📊 Target vocab size: {len(self.tgt_vocab)}")
    
    def _create_default_vocab(self) -> Tuple[List[str], List[str]]:
        """Create a minimal default vocabulary"""
        # Basic English-French vocabulary for demo
        src_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', 
                    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they',
                    'hello', 'world', 'good', 'morning', 'evening', 'night',
                    'cat', 'dog', 'house', 'car', 'book', 'water', 'food',
                    'love', 'like', 'want', 'need', 'go', 'come', 'see', 'know']
        
        tgt_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>',
                    'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont', 'était', 'étaient',
                    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
                    'bonjour', 'monde', 'bon', 'matin', 'soir', 'nuit',
                    'chat', 'chien', 'maison', 'voiture', 'livre', 'eau', 'nourriture',
                    'aimer', 'vouloir', 'avoir', 'besoin', 'aller', 'venir', 'voir', 'savoir']
        
        return src_vocab, tgt_vocab
    
    def translate_with_attention(self, source_sentence: str, max_length: int = 20) -> Dict:
        """
        Translate a sentence and return attention weights
        
        Args:
            source_sentence: Input sentence to translate
            max_length: Maximum translation length
            
        Returns:
            Dictionary containing translation results and attention weights
        """
        # Tokenize and convert to indices
        src_tokens = source_sentence.lower().split()
        src_indices = [self.src_word2id.get(word, self.src_word2id['<UNK>']) for word in src_tokens]
        src_indices = [self.src_word2id['<SOS>']] + src_indices + [self.src_word2id['<EOS>']]
        
        # Convert to tensor
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        src_mask = torch.ones_like(src_tensor, dtype=torch.bool)
        
        with torch.no_grad():
            # Encode
            encoder_outputs, encoder_hidden = self.model.encoder(src_tensor, src_mask)
            
            # Decode with attention tracking
            decoder_input = torch.tensor([[self.tgt_word2id['<SOS>']]], device=self.device)
            decoder_hidden = encoder_hidden
            
            translated_indices = []
            attention_weights = []
            
            for _ in range(max_length):
                decoder_output, decoder_hidden, attention = self.model.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, src_mask
                )
                
                # Get the most likely token
                next_token = decoder_output.argmax(dim=-1)
                translated_indices.append(next_token.item())
                
                # Store attention weights
                if attention is not None:
                    attention_weights.append(attention.squeeze(0).cpu().numpy())
                
                # Check for end of sequence
                if next_token.item() == self.tgt_word2id['<EOS>']:
                    break
                
                # Prepare next input
                decoder_input = next_token.unsqueeze(0)
        
        # Convert indices back to words
        translated_words = [self.tgt_id2word.get(idx, '<UNK>') for idx in translated_indices]
        if translated_words and translated_words[-1] == '<EOS>':
            translated_words = translated_words[:-1]
        
        return {
            'source_tokens': src_tokens,
            'translated_tokens': translated_words,
            'source_indices': src_indices,
            'translated_indices': translated_indices,
            'attention_weights': attention_weights
        }
    
    def plot_attention_heatmap(self, translation_result: Dict, save_path: str = None):
        """
        Visualize attention weights as a heatmap
        
        Args:
            translation_result: Result from translate_with_attention
            save_path: Path to save the plot
        """
        if not translation_result['attention_weights']:
            print("❌ No attention weights available for visualization")
            return
        
        # Prepare data
        src_tokens = ['<SOS>'] + translation_result['source_tokens'] + ['<EOS>']
        tgt_tokens = translation_result['translated_tokens']
        attention_matrix = np.array(translation_result['attention_weights'])
        
        # Ensure attention matrix has correct dimensions
        attention_matrix = attention_matrix[:len(tgt_tokens), :len(src_tokens)]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(8, len(src_tokens) * 0.8), 
                                       max(6, len(tgt_tokens) * 0.6)))
        
        # Create heatmap
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha='right')
        ax.set_yticklabels(tgt_tokens)
        
        # Add text annotations
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                if j < attention_matrix.shape[1]:
                    text = ax.text(j, i, f'{attention_matrix[i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="white" if attention_matrix[i, j] > 0.5 else "black",
                                 fontsize=8)
        
        # Styling
        ax.set_xlabel('Source Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Tokens', fontsize=12, fontweight='bold')
        ax.set_title('Attention Alignment Matrix\n' + 
                    f'"{" ".join(translation_result["source_tokens"])}" → ' +
                    f'"{" ".join(translation_result["translated_tokens"])}"',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved attention heatmap to {save_path}")
        
        plt.show()
    
    def plot_multiple_translations(self, sentences: List[str], save_path: str = None):
        """
        Compare translations for multiple sentences
        
        Args:
            sentences: List of source sentences to translate
            save_path: Path to save the comparison plot
        """
        fig, axes = plt.subplots(len(sentences), 1, 
                               figsize=(12, 4 * len(sentences)))
        if len(sentences) == 1:
            axes = [axes]
        
        for i, sentence in enumerate(sentences):
            result = self.translate_with_attention(sentence)
            
            if result['attention_weights']:
                src_tokens = ['<SOS>'] + result['source_tokens'] + ['<EOS>']
                tgt_tokens = result['translated_tokens']
                attention_matrix = np.array(result['attention_weights'])
                attention_matrix = attention_matrix[:len(tgt_tokens), :len(src_tokens)]
                
                # Create heatmap
                im = axes[i].imshow(attention_matrix, cmap='Blues', aspect='auto')
                
                # Set labels
                axes[i].set_xticks(range(len(src_tokens)))
                axes[i].set_yticks(range(len(tgt_tokens)))
                axes[i].set_xticklabels(src_tokens, rotation=45, ha='right')
                axes[i].set_yticklabels(tgt_tokens)
                
                axes[i].set_title(f'Translation {i+1}: "{sentence}" → "{" ".join(tgt_tokens)}"',
                                fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved comparison plot to {save_path}")
        
        plt.show()
    
    def analyze_attention_patterns(self, sentences: List[str]) -> Dict:
        """
        Analyze attention patterns across multiple translations
        
        Args:
            sentences: List of source sentences
            
        Returns:
            Dictionary with analysis results
        """
        attention_stats = {
            'avg_attention_spread': [],
            'max_attention_values': [],
            'alignment_scores': []
        }
        
        for sentence in sentences:
            result = self.translate_with_attention(sentence)
            
            if result['attention_weights']:
                attention_matrix = np.array(result['attention_weights'])
                
                # Calculate attention spread (entropy-like measure)
                attention_spread = []
                for row in attention_matrix:
                    # Normalize row
                    row_norm = row / (row.sum() + 1e-8)
                    # Calculate entropy
                    entropy = -np.sum(row_norm * np.log(row_norm + 1e-8))
                    attention_spread.append(entropy)
                
                attention_stats['avg_attention_spread'].append(np.mean(attention_spread))
                attention_stats['max_attention_values'].append(np.max(attention_matrix))
                
                # Calculate alignment score (diagonal dominance for monotonic alignment)
                if attention_matrix.shape[0] > 1 and attention_matrix.shape[1] > 1:
                    diagonal_sum = 0
                    total_sum = attention_matrix.sum()
                    min_dim = min(attention_matrix.shape)
                    
                    for i in range(min_dim):
                        if i < attention_matrix.shape[0] and i < attention_matrix.shape[1]:
                            diagonal_sum += attention_matrix[i, i]
                    
                    alignment_score = diagonal_sum / (total_sum + 1e-8)
                    attention_stats['alignment_scores'].append(alignment_score)
        
        return attention_stats
    
    def plot_attention_statistics(self, sentences: List[str], save_path: str = None):
        """
        Plot attention pattern statistics
        
        Args:
            sentences: List of sentences to analyze
            save_path: Path to save the plot
        """
        stats = self.analyze_attention_patterns(sentences)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Attention spread histogram
        if stats['avg_attention_spread']:
            axes[0].hist(stats['avg_attention_spread'], bins=10, alpha=0.7, color='skyblue')
            axes[0].set_xlabel('Average Attention Spread')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Attention Spread\n(Higher = More Distributed)')
            axes[0].grid(True, alpha=0.3)
        
        # Max attention values
        if stats['max_attention_values']:
            axes[1].hist(stats['max_attention_values'], bins=10, alpha=0.7, color='lightcoral')
            axes[1].set_xlabel('Maximum Attention Value')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Max Attention\n(Higher = More Focused)')
            axes[1].grid(True, alpha=0.3)
        
        # Alignment scores
        if stats['alignment_scores']:
            axes[2].hist(stats['alignment_scores'], bins=10, alpha=0.7, color='lightgreen')
            axes[2].set_xlabel('Alignment Score')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Distribution of Alignment Scores\n(Higher = More Monotonic)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved statistics plot to {save_path}")
        
        plt.show()
    
    def interactive_translation(self):
        """Interactive translation demo"""
        print("\n🌐 Interactive Translation Demo")
        print("=" * 50)
        print("Enter English sentences to translate to French (type 'quit' to exit)")
        
        while True:
            sentence = input("\n📝 Enter sentence: ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not sentence:
                continue
            
            try:
                result = self.translate_with_attention(sentence)
                translation = " ".join(result['translated_tokens'])
                
                print(f"🔤 Source: {sentence}")
                print(f"🔄 Translation: {translation}")
                
                # Ask if user wants to see attention visualization
                show_attention = input("📊 Show attention heatmap? (y/n): ").lower().startswith('y')
                if show_attention and result['attention_weights']:
                    self.plot_attention_heatmap(result)
                
            except Exception as e:
                print(f"❌ Error during translation: {e}")

def create_demo_visualizations():
    """Create demonstration visualizations with sample data"""
    print("🎨 Creating demonstration visualizations...")
    
    # Sample sentences for demo
    sample_sentences = [
        "hello world",
        "i love cats",
        "the book is good",
        "we are happy",
        "good morning"
    ]
    
    try:
        # Try to find a trained model
        model_path = None
        checkpoints_dir = "D:/code/github/my_llm/stage3_attention_seq2seq/checkpoints"
        
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
            if checkpoints:
                model_path = os.path.join(checkpoints_dir, checkpoints[0])
        
        if not model_path:
            print("⚠️ No trained model found. Please train a model first using train.py")
            return
        
        # Create visualizer
        visualizer = AttentionVisualizer(model_path)
        
        # Create output directory
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Generating attention visualizations...")
        
        # Single translation example
        result = visualizer.translate_with_attention("hello world")
        visualizer.plot_attention_heatmap(result, 
                                        save_path=os.path.join(output_dir, "attention_demo.png"))
        
        # Multiple translations comparison
        visualizer.plot_multiple_translations(sample_sentences[:3],
                                           save_path=os.path.join(output_dir, "multiple_translations.png"))
        
        # Attention statistics
        visualizer.plot_attention_statistics(sample_sentences,
                                          save_path=os.path.join(output_dir, "attention_stats.png"))
        
        print(f"✅ Visualizations saved to {output_dir}/ directory")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        print("Make sure you have a trained model in the checkpoints directory")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Stage 3 Attention Visualization Tools')
    parser.add_argument('--model_path', type=str, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str,
                       help='Path to vocabulary file (JSON)')
    parser.add_argument('--mode', type=str, choices=['demo', 'interactive', 'analyze'],
                       default='demo', help='Visualization mode')
    parser.add_argument('--sentences', nargs='+',
                       help='Sentences to translate and visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        create_demo_visualizations()
    
    elif args.model_path:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize visualizer
        visualizer = AttentionVisualizer(args.model_path, args.vocab_path)
        
        if args.mode == 'interactive':
            visualizer.interactive_translation()
        
        elif args.mode == 'analyze' and args.sentences:
            print("📊 Analyzing attention patterns...")
            
            # Create visualizations for provided sentences
            for i, sentence in enumerate(args.sentences):
                result = visualizer.translate_with_attention(sentence)
                save_path = os.path.join(args.output_dir, f"attention_{i+1}.png")
                visualizer.plot_attention_heatmap(result, save_path)
            
            # Create comparison plot
            comparison_path = os.path.join(args.output_dir, "comparison.png")
            visualizer.plot_multiple_translations(args.sentences, comparison_path)
            
            # Create statistics plot
            stats_path = os.path.join(args.output_dir, "statistics.png")
            visualizer.plot_attention_statistics(args.sentences, stats_path)
            
            print(f"✅ Analysis complete! Results saved to {args.output_dir}/")
    
    else:
        print("❌ Please provide a model path or use demo mode")
        parser.print_help()

if __name__ == "__main__":
    main()