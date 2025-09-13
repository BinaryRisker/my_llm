"""
Training script for MLP text classification model.

This script demonstrates the complete training pipeline including:
- Data loading and preprocessing
- Model initialization and training
- Evaluation and metrics calculation
- Model saving and checkpointing
"""

import os
import sys
import argparse
import json
import time
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

from models.mlp import SimpleMLP, MLPWithBagOfWords, count_parameters
from utils.data_utils import (
    TextVocabulary, 
    load_ag_news_sample, 
    create_data_loaders
)


class Trainer:
    """
    Trainer class for MLP text classification.
    
    Handles the complete training loop including optimization,
    evaluation, and progress tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        num_classes: int = 4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            # Move data to device
            if 'input_features' in batch:  # Bag-of-words model
                inputs = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
            else:  # Embedding-based model
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = total_correct / total_samples
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # For detailed metrics
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                if 'input_features' in batch:  # Bag-of-words model
                    inputs = batch['input_features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                else:  # Embedding-based model
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Track predictions
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        # Calculate per-class metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        return avg_loss, accuracy, metrics
    
    def _calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict:
        """Calculate detailed classification metrics."""
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        # Calculate precision, recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Calculate macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        return {
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': cm.tolist()
        }
    
    def train(self, num_epochs: int, save_path: str = None) -> Dict:
        """Complete training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    self.save_checkpoint(save_path, epoch, val_acc, val_metrics)
                    print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
        
        # Final training summary
        training_summary = {
            'final_train_loss': self.train_losses[-1],
            'final_train_acc': self.train_accuracies[-1],
            'final_val_loss': self.val_losses[-1],
            'final_val_acc': self.val_accuracies[-1],
            'best_val_acc': best_val_acc,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return training_summary
    
    def save_checkpoint(self, filepath: str, epoch: int, val_acc: float, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'metrics': metrics,
            'train_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        ax1.plot(self.val_losses, label='Validation Loss', marker='s', markersize=3)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', marker='o', markersize=3)
        ax2.plot(self.val_accuracies, label='Validation Accuracy', marker='s', markersize=3)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MLP for text classification')
    parser.add_argument('--model_type', type=str, default='embedding', 
                       choices=['embedding', 'bow'], 
                       help='Model type: embedding or bag-of-words')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128], 
                       help='Hidden layer dimensions')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                       help='Directory to save model checkpoints')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    texts, labels, class_names = load_ag_news_sample()
    
    # Split data (simple split for demonstration)
    split_idx = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    val_texts, val_labels = texts[split_idx:], labels[split_idx:]
    
    print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
    print(f"Classes: {class_names}")
    
    # Build vocabulary
    vocabulary = TextVocabulary(max_vocab_size=args.vocab_size, min_freq=1)
    vocabulary.build_vocabulary(train_texts)
    
    # Create data loaders
    use_bow = (args.model_type == 'bow')
    train_loader, val_loader = create_data_loaders(
        train_texts, train_labels,
        val_texts, val_labels,
        vocabulary,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_bow=use_bow
    )
    
    # Create model
    if args.model_type == 'embedding':
        model = SimpleMLP(
            vocab_size=len(vocabulary),
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            num_classes=len(class_names),
            dropout_rate=args.dropout
        )
    else:  # bag-of-words
        model = MLPWithBagOfWords(
            vocab_size=len(vocabulary),
            hidden_dims=args.hidden_dims,
            num_classes=len(class_names),
            dropout_rate=args.dropout
        )
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_classes=len(class_names)
    )
    
    # Train model
    checkpoint_path = os.path.join(args.save_dir, f'best_{args.model_type}_mlp.pt')
    training_summary = trainer.train(args.epochs, checkpoint_path)
    
    # Save vocabulary and training config
    vocab_path = os.path.join(args.save_dir, 'vocabulary.pkl')
    vocabulary.save(vocab_path)
    
    config_path = os.path.join(args.save_dir, 'config.json')
    config = {
        'model_type': args.model_type,
        'vocab_size': len(vocabulary),
        'embedding_dim': args.embedding_dim,
        'hidden_dims': args.hidden_dims,
        'num_classes': len(class_names),
        'class_names': class_names,
        'max_length': args.max_length,
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
    
    print(f"\n📁 All files saved to: {args.save_dir}")
    print("Files created:")
    print(f"  - Model checkpoint: {checkpoint_path}")
    print(f"  - Vocabulary: {vocab_path}")
    print(f"  - Configuration: {config_path}")
    print(f"  - Training plots: {plot_path}")
    print(f"  - Training summary: {summary_path}")


if __name__ == '__main__':
    main()
