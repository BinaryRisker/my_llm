"""
Training script for RNN/LSTM text generation models.

This script demonstrates the complete training pipeline for language modeling
including teacher forcing, gradient clipping, and various sampling strategies.
"""

import os
import sys
import argparse
import json
import time
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.rnn import SimpleRNN, create_rnn_model
from models.lstm import SimpleLSTM, BiLSTM, count_parameters
from utils.text_data import (
    CharacterVocabulary, 
    WordVocabulary,
    create_sample_text,
    split_text_data,
    create_data_loaders,
    calculate_perplexity
)


class LanguageModelTrainer:
    """
    Trainer class for RNN/LSTM language models.
    
    Handles training with teacher forcing, gradient clipping,
    and various optimization strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocabulary,
        device: torch.device,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        teacher_forcing_ratio: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_seq = batch['input'].to(self.device)  # [batch_size, seq_len]
            target_seq = batch['target'].to(self.device)  # [batch_size, seq_len]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'init_hidden'):
                # For custom RNN/LSTM models
                hidden = self.model.init_hidden(input_seq.size(0), self.device)
                output, _ = self.model(input_seq, hidden)
            else:
                # For PyTorch built-in models
                output, _ = self.model(input_seq)
            
            # Calculate loss
            # output: [batch_size, seq_len, vocab_size]
            # target: [batch_size, seq_len]
            loss = self.criterion(
                output.view(-1, output.size(-1)), 
                target_seq.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_tokens += target_seq.numel()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            perplexity = calculate_perplexity(avg_loss)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'ppl': f'{perplexity:.2f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        perplexity = calculate_perplexity(avg_loss)
        
        return avg_loss, perplexity
    
    def validate(self) -> Tuple[float, float]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_seq = batch['input'].to(self.device)
                target_seq = batch['target'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'init_hidden'):
                    hidden = self.model.init_hidden(input_seq.size(0), self.device)
                    output, _ = self.model(input_seq, hidden)
                else:
                    output, _ = self.model(input_seq)
                
                # Calculate loss
                loss = self.criterion(
                    output.view(-1, output.size(-1)), 
                    target_seq.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += target_seq.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = calculate_perplexity(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self, num_epochs: int, save_path: str = None) -> Dict:
        """Complete training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            start_time = time.time()
            train_loss, train_ppl = self.train_epoch()
            
            # Validate
            val_loss, val_ppl = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_ppl)
            self.val_perplexities.append(val_ppl)
            
            # Calculate time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Time: {epoch_time:.2f}s")
            
            # Generate sample text
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.generate_sample()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_loss)
                    print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
        
        # Final training summary
        training_summary = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_ppl': self.train_perplexities[-1],
            'final_val_ppl': self.val_perplexities[-1],
            'best_val_loss': best_val_loss,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_perplexities': self.train_perplexities,
                'val_perplexities': self.val_perplexities
            }
        }
        
        print(f"\\n🎉 Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")
        
        return training_summary
    
    def generate_sample(self, max_length: int = 200, temperature: float = 0.8):
        """Generate sample text to monitor training progress."""
        self.model.eval()
        
        # Start with a random character/word
        if hasattr(self.vocabulary, 'char2idx'):
            start_token = self.vocabulary.char2idx.get('T', 1)
            vocab_type = "char"
        else:
            start_token = self.vocabulary.word2idx.get('<START>', 1)
            vocab_type = "word"
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                # Custom models with generate method
                generated = self.model.generate(
                    start_token=start_token,
                    max_length=max_length,
                    temperature=temperature,
                    device=self.device
                )
            else:
                # Built-in PyTorch models - simple generation
                generated = [start_token]
                hidden = None
                
                for _ in range(max_length - 1):
                    input_tensor = torch.tensor([[generated[-1]]], device=self.device)
                    output, hidden = self.model(input_tensor, hidden)
                    
                    logits = output[0, 0, :] / temperature
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, 1).item()
                    generated.append(next_token)
            
            # Convert to text
            generated_text = self.vocabulary.indices_to_text(generated)
            print(f"\\n📝 Generated sample ({vocab_type}-level):")
            print(f"'{generated_text[:150]}...'")
    
    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_perplexities': self.train_perplexities,
                'val_perplexities': self.val_perplexities
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', marker='s', markersize=3)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Perplexity plot
        ax2.plot(epochs, self.train_perplexities, 'b-', label='Train Perplexity', marker='o', markersize=3)
        ax2.plot(epochs, self.val_perplexities, 'r-', label='Validation Perplexity', marker='s', markersize=3)
        ax2.set_title('Training and Validation Perplexity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Learning curve analysis
        if len(self.train_losses) > 1:
            train_improvement = np.diff(self.train_losses)
            ax3.plot(epochs[1:], train_improvement, 'g-', marker='v', markersize=3)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Training Loss Improvement')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss Change')
            ax3.grid(True, alpha=0.3)
            
            # Overfitting analysis
            train_val_diff = np.array(self.val_losses) - np.array(self.train_losses)
            ax4.plot(epochs, train_val_diff, 'purple', marker='d', markersize=3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Overfitting Analysis (Val - Train Loss)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Difference')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train RNN/LSTM for text generation')
    parser.add_argument('--model_type', type=str, default='lstm', 
                       choices=['rnn', 'lstm', 'gru'],
                       help='Model type')
    parser.add_argument('--vocab_type', type=str, default='char',
                       choices=['char', 'word'],
                       help='Vocabulary type: character or word level')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--seq_length', type=int, default=64,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden state dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                       help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3, 
                       help='Dropout rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                       help='Teacher forcing ratio')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to text file for training (optional)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create data
    print("Loading data...")
    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        # Split into sentences or paragraphs for training
        texts = [line.strip() for line in text_content.split('\n') if line.strip()]
    else:
        texts = create_sample_text()
        print("Using sample text data")
    
    # Split data
    train_texts, val_texts, test_texts = split_text_data(texts)
    print(f"Data split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Build vocabulary
    print(f"Building {args.vocab_type}-level vocabulary...")
    if args.vocab_type == 'char':
        vocabulary = CharacterVocabulary()
    else:
        vocabulary = WordVocabulary(max_vocab_size=10000, min_freq=1)
    
    vocabulary.build_vocabulary(train_texts)
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_texts, val_texts,
        vocabulary,
        seq_length=args.seq_length,
        batch_size=args.batch_size
    )
    
    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    if args.model_type == 'lstm':
        model = SimpleLSTM(
            vocab_size=len(vocabulary),
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = create_rnn_model(
            args.model_type,
            vocab_size=len(vocabulary),
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = LanguageModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocabulary,
        device=device,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        teacher_forcing_ratio=args.teacher_forcing_ratio
    )
    
    # Train model
    checkpoint_path = os.path.join(args.save_dir, f'best_{args.model_type}_{args.vocab_type}.pt')
    training_summary = trainer.train(args.epochs, checkpoint_path)
    
    # Save vocabulary and config
    vocab_path = os.path.join(args.save_dir, f'{args.vocab_type}_vocabulary.pkl')
    vocabulary.save(vocab_path)
    
    config_path = os.path.join(args.save_dir, 'config.json')
    config = {
        'model_type': args.model_type,
        'vocab_type': args.vocab_type,
        'vocab_size': len(vocabulary),
        'embedding_dim': args.embedding_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'seq_length': args.seq_length,
        'dropout': args.dropout
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plot training history
    plot_path = os.path.join(args.save_dir, 'training_history.png')
    trainer.plot_training_history(plot_path)
    
    # Save training summary
    summary_path = os.path.join(args.save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\\n📁 All files saved to: {args.save_dir}")
    print("Files created:")
    print(f"  - Model checkpoint: {checkpoint_path}")
    print(f"  - Vocabulary: {vocab_path}")
    print(f"  - Configuration: {config_path}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - Training summary: {summary_path}")
    
    # Final generation test
    print("\\n🎭 Final text generation test:")
    trainer.generate_sample(max_length=300, temperature=0.7)


if __name__ == '__main__':
    main()