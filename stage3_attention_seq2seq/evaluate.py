#!/usr/bin/env python3
"""
Stage 3: Attention Mechanism and Seq2Seq - Evaluation Tools
===========================================================

This script provides comprehensive evaluation tools for attention-based seq2seq models:
- BLEU score computation
- Translation quality assessment
- Model comparison across different attention mechanisms
- Performance benchmarking
- Error analysis and visualization

Author: AI Assistant
Date: 2024
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Seq2SeqModel
from utils.data_loader import TranslationDataset, create_vocab, load_translation_data
from utils.bleu_eval import compute_bleu, compute_corpus_bleu
from utils.training import translate_sentence

class Seq2SeqEvaluator:
    """Comprehensive evaluation toolkit for Seq2Seq models"""
    
    def __init__(self, model_path: str, vocab_path: str = None, data_path: str = None):
        """
        Initialize evaluator with model and data
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            data_path: Path to test data
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = checkpoint['model']
        self.model.eval()
        
        # Load vocabulary
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.src_vocab = vocab_data['src_vocab']
                self.tgt_vocab = vocab_data['tgt_vocab']
        else:
            print("Warning: Using default vocabulary")
            self.src_vocab, self.tgt_vocab = self._create_default_vocab()
        
        # Create word mappings
        self.src_word2id = {word: i for i, word in enumerate(self.src_vocab)}
        self.tgt_word2id = {word: i for i, word in enumerate(self.tgt_vocab)}
        self.src_id2word = {i: word for word, i in self.src_word2id.items()}
        self.tgt_id2word = {i: word for word, i in self.tgt_word2id.items()}
        
        # Load test data if provided
        self.test_data = []
        if data_path and os.path.exists(data_path):
            self.test_data = self._load_test_data(data_path)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"📊 Source vocab: {len(self.src_vocab)} words")
        print(f"📊 Target vocab: {len(self.tgt_vocab)} words")
        print(f"📊 Test samples: {len(self.test_data)}")
    
    def _create_default_vocab(self) -> Tuple[List[str], List[str]]:
        """Create default vocabulary for testing"""
        src_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', 
                    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that',
                    'hello', 'world', 'good', 'morning', 'evening', 'night', 'today',
                    'cat', 'dog', 'house', 'car', 'book', 'water', 'food', 'time',
                    'love', 'like', 'want', 'need', 'go', 'come', 'see', 'know', 'think']
        
        tgt_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>',
                    'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont', 'était', 'étaient',
                    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'ce', 'cette',
                    'bonjour', 'monde', 'bon', 'matin', 'soir', 'nuit', 'aujourd', 'hui',
                    'chat', 'chien', 'maison', 'voiture', 'livre', 'eau', 'nourriture', 'temps',
                    'aimer', 'vouloir', 'avoir', 'besoin', 'aller', 'venir', 'voir', 'savoir', 'penser']
        
        return src_vocab, tgt_vocab
    
    def _load_test_data(self, data_path: str) -> List[Tuple[str, str]]:
        """Load test data from file"""
        test_pairs = []
        
        try:
            if data_path.endswith('.json'):
                # JSON format
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'source' in item and 'target' in item:
                            test_pairs.append((item['source'], item['target']))
            
            else:
                # Plain text format (assuming tab-separated)
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if '\t' in line:
                            src, tgt = line.split('\t', 1)
                            test_pairs.append((src, tgt))
                        elif '|' in line:
                            src, tgt = line.split('|', 1)
                            test_pairs.append((src, tgt))
        
        except Exception as e:
            print(f"Warning: Could not load test data from {data_path}: {e}")
            # Create sample test data
            test_pairs = [
                ("hello world", "bonjour monde"),
                ("i love cats", "j'aime les chats"),
                ("the book is good", "le livre est bon"),
                ("we are happy", "nous sommes heureux"),
                ("good morning", "bonjour"),
                ("how are you", "comment allez vous"),
                ("thank you", "merci"),
                ("see you later", "à bientôt")
            ]
        
        return test_pairs
    
    def translate_sentence(self, source_sentence: str, max_length: int = 20, 
                          beam_size: int = 1) -> List[str]:
        """
        Translate a single sentence
        
        Args:
            source_sentence: Input sentence
            max_length: Maximum translation length
            beam_size: Beam search size (1 = greedy)
            
        Returns:
            List of translated tokens
        """
        # Tokenize and convert to indices
        src_tokens = source_sentence.lower().strip().split()
        src_indices = [self.src_word2id.get(word, self.src_word2id['<UNK>']) for word in src_tokens]
        src_indices = [self.src_word2id['<SOS>']] + src_indices + [self.src_word2id['<EOS>']]
        
        # Convert to tensor
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        src_mask = torch.ones_like(src_tensor, dtype=torch.bool)
        
        with torch.no_grad():
            # Encode
            encoder_outputs, encoder_hidden = self.model.encoder(src_tensor, src_mask)
            
            if beam_size == 1:
                # Greedy decoding
                translated_indices = self._greedy_decode(encoder_outputs, encoder_hidden, 
                                                       src_mask, max_length)
            else:
                # Beam search decoding
                translated_indices = self._beam_search_decode(encoder_outputs, encoder_hidden, 
                                                            src_mask, max_length, beam_size)
        
        # Convert indices back to words
        translated_words = []
        for idx in translated_indices:
            if idx == self.tgt_word2id['<EOS>']:
                break
            word = self.tgt_id2word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<SOS>']:
                translated_words.append(word)
        
        return translated_words
    
    def _greedy_decode(self, encoder_outputs, encoder_hidden, src_mask, max_length):
        """Greedy decoding implementation"""
        decoder_input = torch.tensor([[self.tgt_word2id['<SOS>']]], device=self.device)
        decoder_hidden = encoder_hidden
        
        translated_indices = []
        
        for _ in range(max_length):
            decoder_output, decoder_hidden, _ = self.model.decoder(
                decoder_input, decoder_hidden, encoder_outputs, src_mask
            )
            
            next_token = decoder_output.argmax(dim=-1)
            translated_indices.append(next_token.item())
            
            if next_token.item() == self.tgt_word2id['<EOS>']:
                break
            
            decoder_input = next_token.unsqueeze(0)
        
        return translated_indices
    
    def _beam_search_decode(self, encoder_outputs, encoder_hidden, src_mask, 
                           max_length, beam_size):
        """Beam search decoding implementation"""
        # Initialize beam
        beams = [(0.0, [self.tgt_word2id['<SOS>']], encoder_hidden)]
        
        for step in range(max_length):
            new_beams = []
            
            for score, sequence, hidden in beams:
                if sequence[-1] == self.tgt_word2id['<EOS>']:
                    new_beams.append((score, sequence, hidden))
                    continue
                
                # Decode next token
                decoder_input = torch.tensor([[sequence[-1]]], device=self.device)
                decoder_output, new_hidden, _ = self.model.decoder(
                    decoder_input, hidden, encoder_outputs, src_mask
                )
                
                # Get top-k candidates
                log_probs = torch.log_softmax(decoder_output, dim=-1).squeeze(0).squeeze(0)
                top_scores, top_indices = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_score = score + top_scores[i].item()
                    new_sequence = sequence + [top_indices[i].item()]
                    new_beams.append((new_score, new_sequence, new_hidden))
            
            # Keep only top beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            
            # Check if all beams ended
            if all(seq[-1] == self.tgt_word2id['<EOS>'] for _, seq, _ in beams):
                break
        
        # Return best sequence
        return beams[0][1][1:]  # Skip initial <SOS>
    
    def compute_bleu_scores(self, test_pairs: List[Tuple[str, str]] = None,
                           beam_size: int = 1) -> Dict:
        """
        Compute BLEU scores for test data
        
        Args:
            test_pairs: List of (source, target) pairs
            beam_size: Beam search size
            
        Returns:
            Dictionary with BLEU scores and statistics
        """
        if test_pairs is None:
            test_pairs = self.test_data
        
        if not test_pairs:
            print("❌ No test data available")
            return {}
        
        print(f"🔄 Computing BLEU scores for {len(test_pairs)} samples...")
        
        translations = []
        references = []
        individual_bleu_scores = []
        
        start_time = time.time()
        
        for i, (source, target) in enumerate(test_pairs):
            # Translate
            translated_tokens = self.translate_sentence(source, beam_size=beam_size)
            translation = ' '.join(translated_tokens)
            
            # Compute individual BLEU
            reference_tokens = target.lower().strip().split()
            bleu_score = compute_bleu([translated_tokens], [reference_tokens])
            
            translations.append(translation)
            references.append(target)
            individual_bleu_scores.append(bleu_score)
            
            if (i + 1) % 10 == 0:
                print(f"📊 Processed {i + 1}/{len(test_pairs)} samples")
        
        # Compute corpus-level BLEU
        all_translated_tokens = [trans.split() for trans in translations]
        all_reference_tokens = [ref.lower().strip().split() for ref in references]
        corpus_bleu = compute_corpus_bleu(all_translated_tokens, all_reference_tokens)
        
        processing_time = time.time() - start_time
        
        results = {
            'individual_bleu_scores': individual_bleu_scores,
            'corpus_bleu': corpus_bleu,
            'avg_bleu': np.mean(individual_bleu_scores),
            'std_bleu': np.std(individual_bleu_scores),
            'min_bleu': np.min(individual_bleu_scores),
            'max_bleu': np.max(individual_bleu_scores),
            'translations': translations,
            'references': references,
            'processing_time': processing_time,
            'samples_per_second': len(test_pairs) / processing_time
        }
        
        return results
    
    def analyze_translation_quality(self, results: Dict) -> Dict:
        """
        Analyze translation quality beyond BLEU scores
        
        Args:
            results: Results from compute_bleu_scores
            
        Returns:
            Quality analysis metrics
        """
        translations = results['translations']
        references = results['references']
        
        analysis = {
            'length_ratio': [],
            'vocabulary_diversity': [],
            'repeated_words': [],
            'exact_matches': 0,
            'empty_translations': 0
        }
        
        for translation, reference in zip(translations, references):
            trans_words = translation.split()
            ref_words = reference.split()
            
            # Length ratio
            length_ratio = len(trans_words) / (len(ref_words) + 1e-8)
            analysis['length_ratio'].append(length_ratio)
            
            # Vocabulary diversity (unique words / total words)
            if trans_words:
                diversity = len(set(trans_words)) / len(trans_words)
            else:
                diversity = 0.0
                analysis['empty_translations'] += 1
            analysis['vocabulary_diversity'].append(diversity)
            
            # Repeated words percentage
            if trans_words:
                word_counts = {}
                for word in trans_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                repeated = sum(count - 1 for count in word_counts.values() if count > 1)
                repeated_ratio = repeated / len(trans_words)
            else:
                repeated_ratio = 0.0
            analysis['repeated_words'].append(repeated_ratio)
            
            # Exact matches
            if translation.lower().strip() == reference.lower().strip():
                analysis['exact_matches'] += 1
        
        # Compute summary statistics
        analysis['avg_length_ratio'] = np.mean(analysis['length_ratio'])
        analysis['avg_vocabulary_diversity'] = np.mean(analysis['vocabulary_diversity'])
        analysis['avg_repeated_words'] = np.mean(analysis['repeated_words'])
        analysis['exact_match_rate'] = analysis['exact_matches'] / len(translations)
        analysis['empty_translation_rate'] = analysis['empty_translations'] / len(translations)
        
        return analysis
    
    def plot_evaluation_results(self, results: Dict, analysis: Dict = None, 
                              save_path: str = None):
        """
        Visualize evaluation results
        
        Args:
            results: BLEU score results
            analysis: Quality analysis results
            save_path: Path to save the plot
        """
        if analysis is None:
            analysis = self.analyze_translation_quality(results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Seq2Seq Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # BLEU score distribution
        axes[0, 0].hist(results['individual_bleu_scores'], bins=20, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[0, 0].axvline(results['avg_bleu'], color='red', linestyle='--', 
                          label=f'Mean: {results["avg_bleu"]:.3f}')
        axes[0, 0].set_xlabel('BLEU Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('BLEU Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Length ratio distribution
        axes[0, 1].hist(analysis['length_ratio'], bins=20, alpha=0.7, 
                       color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(1.0, color='green', linestyle='-', alpha=0.7, 
                          label='Perfect ratio')
        axes[0, 1].axvline(analysis['avg_length_ratio'], color='red', linestyle='--',
                          label=f'Mean: {analysis["avg_length_ratio"]:.3f}')
        axes[0, 1].set_xlabel('Translation Length / Reference Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Length Ratio Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Vocabulary diversity
        axes[0, 2].hist(analysis['vocabulary_diversity'], bins=20, alpha=0.7,
                       color='lightgreen', edgecolor='black')
        axes[0, 2].axvline(analysis['avg_vocabulary_diversity'], color='red', linestyle='--',
                          label=f'Mean: {analysis["avg_vocabulary_diversity"]:.3f}')
        axes[0, 2].set_xlabel('Vocabulary Diversity (Unique/Total Words)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Vocabulary Diversity Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # BLEU vs Length ratio scatter
        axes[1, 0].scatter(analysis['length_ratio'], results['individual_bleu_scores'],
                          alpha=0.6, color='purple')
        axes[1, 0].set_xlabel('Length Ratio')
        axes[1, 0].set_ylabel('BLEU Score')
        axes[1, 0].set_title('BLEU Score vs Length Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Quality metrics bar chart
        quality_metrics = {
            'Corpus BLEU': results['corpus_bleu'],
            'Avg BLEU': results['avg_bleu'],
            'Exact Match Rate': analysis['exact_match_rate'],
            'Avg Vocab Diversity': analysis['avg_vocabulary_diversity'],
            'Empty Translation Rate': analysis['empty_translation_rate']
        }
        
        bars = axes[1, 1].bar(range(len(quality_metrics)), list(quality_metrics.values()),
                             color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'])
        axes[1, 1].set_xticks(range(len(quality_metrics)))
        axes[1, 1].set_xticklabels(list(quality_metrics.keys()), rotation=45, ha='right')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Quality Metrics Summary')
        
        # Add value labels on bars
        for bar, value in zip(bars, quality_metrics.values()):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Performance metrics
        perf_text = f"""Performance Statistics:
        
Total Samples: {len(results['individual_bleu_scores'])}
Processing Time: {results['processing_time']:.2f}s
Samples/Second: {results['samples_per_second']:.2f}

BLEU Statistics:
Mean ± Std: {results['avg_bleu']:.3f} ± {results['std_bleu']:.3f}
Min / Max: {results['min_bleu']:.3f} / {results['max_bleu']:.3f}
Corpus BLEU: {results['corpus_bleu']:.3f}

Quality Statistics:
Exact Matches: {analysis['exact_matches']}/{len(results['translations'])} ({analysis['exact_match_rate']:.1%})
Empty Translations: {analysis['empty_translations']}/{len(results['translations'])} ({analysis['empty_translation_rate']:.1%})
Avg Length Ratio: {analysis['avg_length_ratio']:.3f}
Avg Vocab Diversity: {analysis['avg_vocabulary_diversity']:.3f}"""
        
        axes[1, 2].text(0.05, 0.95, perf_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=9)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved evaluation plot to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict, analysis: Dict = None,
                                 save_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            results: BLEU score results
            analysis: Quality analysis results
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if analysis is None:
            analysis = self.analyze_translation_quality(results)
        
        report = f"""
# Seq2Seq Model Evaluation Report

## Model Performance Summary

### BLEU Scores
- **Corpus BLEU**: {results['corpus_bleu']:.4f}
- **Average BLEU**: {results['avg_bleu']:.4f} ± {results['std_bleu']:.4f}
- **BLEU Range**: {results['min_bleu']:.4f} - {results['max_bleu']:.4f}

### Translation Quality Metrics
- **Exact Match Rate**: {analysis['exact_match_rate']:.2%} ({analysis['exact_matches']}/{len(results['translations'])})
- **Empty Translation Rate**: {analysis['empty_translation_rate']:.2%} ({analysis['empty_translations']}/{len(results['translations'])})
- **Average Length Ratio**: {analysis['avg_length_ratio']:.3f}
- **Average Vocabulary Diversity**: {analysis['avg_vocabulary_diversity']:.3f}
- **Average Repeated Words**: {analysis['avg_repeated_words']:.3f}

### Performance Metrics
- **Total Test Samples**: {len(results['individual_bleu_scores'])}
- **Processing Time**: {results['processing_time']:.2f} seconds
- **Translation Speed**: {results['samples_per_second']:.2f} samples/second

## Sample Translations

### Best Translations (Top 5 BLEU Scores)
"""
        # Add best translations
        best_indices = np.argsort(results['individual_bleu_scores'])[-5:][::-1]
        for i, idx in enumerate(best_indices, 1):
            bleu = results['individual_bleu_scores'][idx]
            source = self.test_data[idx][0] if idx < len(self.test_data) else "N/A"
            reference = results['references'][idx]
            translation = results['translations'][idx]
            
            report += f"""
{i}. **BLEU: {bleu:.4f}**
   - Source: "{source}"
   - Reference: "{reference}"
   - Translation: "{translation}"
"""
        
        report += "\n### Worst Translations (Bottom 5 BLEU Scores)\n"
        
        # Add worst translations
        worst_indices = np.argsort(results['individual_bleu_scores'])[:5]
        for i, idx in enumerate(worst_indices, 1):
            bleu = results['individual_bleu_scores'][idx]
            source = self.test_data[idx][0] if idx < len(self.test_data) else "N/A"
            reference = results['references'][idx]
            translation = results['translations'][idx]
            
            report += f"""
{i}. **BLEU: {bleu:.4f}**
   - Source: "{source}"
   - Reference: "{reference}"
   - Translation: "{translation}"
"""
        
        report += f"""

## Recommendations

### Model Performance
- {'✅' if results['corpus_bleu'] > 0.20 else '⚠️'} Corpus BLEU score: {results['corpus_bleu']:.4f} ({'Good' if results['corpus_bleu'] > 0.20 else 'Needs improvement'})
- {'✅' if analysis['exact_match_rate'] > 0.05 else '⚠️'} Exact match rate: {analysis['exact_match_rate']:.2%} ({'Acceptable' if analysis['exact_match_rate'] > 0.05 else 'Low'})
- {'✅' if analysis['empty_translation_rate'] < 0.10 else '⚠️'} Empty translation rate: {analysis['empty_translation_rate']:.2%} ({'Good' if analysis['empty_translation_rate'] < 0.10 else 'High'})

### Potential Improvements
"""
        
        # Add improvement suggestions
        if results['corpus_bleu'] < 0.15:
            report += "- 📈 Consider increasing model capacity or training longer\n"
        if analysis['avg_length_ratio'] < 0.7:
            report += "- 📏 Translations tend to be too short - adjust length penalties\n"
        if analysis['avg_length_ratio'] > 1.5:
            report += "- 📏 Translations tend to be too long - increase length penalties\n"
        if analysis['avg_repeated_words'] > 0.2:
            report += "- 🔄 High repetition rate - consider coverage mechanisms\n"
        if analysis['empty_translation_rate'] > 0.05:
            report += "- 🚫 Too many empty translations - check vocabulary coverage\n"
        
        report += f"""
---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Saved evaluation report to {save_path}")
        
        return report

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Stage 3 Seq2Seq Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab_path', type=str,
                       help='Path to vocabulary file (JSON)')
    parser.add_argument('--test_data', type=str,
                       help='Path to test data file')
    parser.add_argument('--beam_size', type=int, default=1,
                       help='Beam search size (1 = greedy)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--save_translations', action='store_true',
                       help='Save all translations to file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Starting Seq2Seq model evaluation...")
    print(f"📂 Model: {args.model_path}")
    print(f"📂 Vocab: {args.vocab_path or 'Default'}")
    print(f"📂 Test data: {args.test_data or 'Default'}")
    print(f"🔍 Beam size: {args.beam_size}")
    
    # Initialize evaluator
    evaluator = Seq2SeqEvaluator(args.model_path, args.vocab_path, args.test_data)
    
    # Compute BLEU scores
    results = evaluator.compute_bleu_scores(beam_size=args.beam_size)
    
    if not results:
        print("❌ Evaluation failed - no results generated")
        return
    
    # Analyze translation quality
    analysis = evaluator.analyze_translation_quality(results)
    
    # Generate visualizations
    plot_path = os.path.join(args.output_dir, 'evaluation_results.png')
    evaluator.plot_evaluation_results(results, analysis, plot_path)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.md')
    report = evaluator.generate_evaluation_report(results, analysis, report_path)
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, 'detailed_results.json')
    detailed_results = {
        'bleu_results': results,
        'quality_analysis': analysis,
        'model_path': args.model_path,
        'beam_size': args.beam_size,
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save translations if requested
    if args.save_translations:
        trans_path = os.path.join(args.output_dir, 'translations.txt')
        with open(trans_path, 'w', encoding='utf-8') as f:
            f.write("Source\tReference\tTranslation\tBLEU\n")
            for i, (source, ref, trans, bleu) in enumerate(zip(
                [pair[0] for pair in evaluator.test_data],
                results['references'],
                results['translations'],
                results['individual_bleu_scores']
            )):
                f.write(f"{source}\t{ref}\t{trans}\t{bleu:.4f}\n")
        print(f"💾 Saved translations to {trans_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"📈 Corpus BLEU: {results['corpus_bleu']:.4f}")
    print(f"📈 Average BLEU: {results['avg_bleu']:.4f} ± {results['std_bleu']:.4f}")
    print(f"🎯 Exact matches: {analysis['exact_matches']}/{len(results['translations'])} ({analysis['exact_match_rate']:.1%})")
    print(f"⚡ Speed: {results['samples_per_second']:.2f} samples/second")
    print(f"📁 Results saved to: {args.output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()