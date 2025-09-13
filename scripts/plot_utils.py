"""
Universal Plotting Utilities
============================

This module provides common plotting functions that can be used across
all stages of the project for visualization and analysis.

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_curves(history: Dict[str, List], 
                        save_path: Optional[str] = None,
                        title: str = "Training Progress",
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot comprehensive training curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(history.get('train_loss', []), label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Perplexity curves
    if 'train_perplexity' in history:
        axes[0, 1].plot(history['train_perplexity'], label='Train PPL', linewidth=2)
        if 'val_perplexity' in history and history['val_perplexity']:
            axes[0, 1].plot(history['val_perplexity'], label='Val PPL', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Perplexity Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # Learning rate schedule
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'], linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Gradient norms (if available)
    if 'gradient_norm' in history:
        axes[1, 1].plot(history['gradient_norm'], linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Show training summary instead
        train_summary = f"""
        Training Summary:
        
        Total Epochs: {len(history.get('train_loss', []))}
        Final Train Loss: {history.get('train_loss', [0])[-1]:.4f}
        """
        if 'val_loss' in history and history['val_loss']:
            train_summary += f"Final Val Loss: {history['val_loss'][-1]:.4f}\n"
        if 'train_perplexity' in history:
            train_summary += f"Final Train PPL: {history['train_perplexity'][-1]:.2f}\n"
        
        axes[1, 1].text(0.1, 0.5, train_summary, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
    
    plt.show()

def plot_attention_matrix(attention_weights: np.ndarray,
                         source_tokens: List[str],
                         target_tokens: List[str],
                         save_path: Optional[str] = None,
                         title: str = "Attention Matrix",
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot attention weight matrix as heatmap
    
    Args:
        attention_weights: Attention weights [tgt_len, src_len]
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha='right')
    ax.set_yticklabels(target_tokens)
    
    # Add text annotations
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            if j < attention_weights.shape[1] and i < attention_weights.shape[0]:
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if attention_weights[i, j] > 0.5 else "black",
                             fontsize=8)
    
    # Styling
    ax.set_xlabel('Source Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Tokens', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Attention matrix saved to {save_path}")
    
    plt.show()

def plot_model_comparison(results: Dict[str, Dict],
                         metrics: List[str] = ['bleu', 'loss', 'perplexity'],
                         save_path: Optional[str] = None,
                         title: str = "Model Comparison",
                         figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Compare multiple models across different metrics
    
    Args:
        results: Dictionary mapping model names to their results
        metrics: List of metrics to compare
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    model_names = list(results.keys())
    
    for i, metric in enumerate(metrics):
        values = []
        valid_models = []
        
        for model_name in model_names:
            if metric in results[model_name]:
                values.append(results[model_name][metric])
                valid_models.append(model_name)
        
        if values:
            bars = axes[i].bar(range(len(valid_models)), values, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(valid_models))))
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_xticks(range(len(valid_models)))
            axes[i].set_xticklabels(valid_models, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3, axis='y')
        else:
            axes[i].text(0.5, 0.5, f'No {metric} data available',
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{metric.upper()} Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison saved to {save_path}")
    
    plt.show()

def plot_generation_examples(examples: List[Dict],
                           save_path: Optional[str] = None,
                           title: str = "Generation Examples",
                           max_examples: int = 5) -> None:
    """
    Plot text generation examples
    
    Args:
        examples: List of generation examples, each with 'input', 'target', 'generated'
        save_path: Path to save the plot
        title: Plot title
        max_examples: Maximum number of examples to show
    """
    fig, ax = plt.subplots(figsize=(15, min(len(examples), max_examples) * 2))
    
    examples = examples[:max_examples]
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(examples))
    
    for i, example in enumerate(examples):
        y_pos = len(examples) - i - 1
        
        # Input text
        input_text = example.get('input', 'N/A')
        ax.text(0.5, y_pos + 0.7, f"Input: {input_text}", 
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Target text (if available)
        if 'target' in example:
            target_text = example['target']
            ax.text(0.5, y_pos + 0.35, f"Target: {target_text}", 
                   fontsize=10, color='green',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Generated text
        generated_text = example.get('generated', 'N/A')
        ax.text(0.5, y_pos, f"Generated: {generated_text}", 
               fontsize=10, color='red',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Generation examples saved to {save_path}")
    
    plt.show()

def plot_loss_landscape(loss_surface: np.ndarray,
                       param1_range: np.ndarray,
                       param2_range: np.ndarray,
                       param1_name: str = "Parameter 1",
                       param2_name: str = "Parameter 2",
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot loss landscape (2D surface plot)
    
    Args:
        loss_surface: 2D array of loss values
        param1_range: Parameter 1 values
        param2_range: Parameter 2 values
        param1_name: Name of parameter 1
        param2_name: Name of parameter 2
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    P1, P2 = np.meshgrid(param1_range, param2_range)
    
    # Create surface plot
    surf = ax.plot_surface(P1, P2, loss_surface, cmap='viridis', alpha=0.8)
    
    # Create contour plot
    contours = ax.contour(P1, P2, loss_surface, zdir='z', offset=loss_surface.min(), 
                         cmap='viridis', alpha=0.5)
    
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Loss landscape saved to {save_path}")
    
    plt.show()

def plot_embeddings_2d(embeddings: np.ndarray,
                      labels: Optional[List[str]] = None,
                      method: str = 'tsne',
                      save_path: Optional[str] = None,
                      title: str = "Embeddings Visualization",
                      figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot 2D embedding visualization using t-SNE or PCA
    
    Args:
        embeddings: Embedding matrix [n_samples, n_features]
        labels: Optional labels for each embedding
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    # Dimensionality reduction
    if method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           alpha=0.6, s=50, c=range(len(embeddings_2d)), cmap='viridis')
    else:
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      label=label, alpha=0.6, s=50, c=[colors[i]])
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.set_title(f'{title} ({method.upper()})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Embeddings visualization saved to {save_path}")
    
    plt.show()

def create_training_dashboard(history: Dict[str, List],
                            attention_data: Optional[Dict] = None,
                            examples: Optional[List[Dict]] = None,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (20, 15)) -> None:
    """
    Create comprehensive training dashboard
    
    Args:
        history: Training history
        attention_data: Optional attention visualization data
        examples: Optional generation examples
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Determine subplot layout
    n_rows = 3
    n_cols = 3
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Training Dashboard', fontsize=20, fontweight='bold')
    
    # Training curves
    ax1 = plt.subplot(n_rows, n_cols, 1)
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Perplexity curves
    ax2 = plt.subplot(n_rows, n_cols, 2)
    if 'train_perplexity' in history:
        ax2.plot(history['train_perplexity'], label='Train PPL', linewidth=2)
    if 'val_perplexity' in history and history['val_perplexity']:
        ax2.plot(history['val_perplexity'], label='Val PPL', linewidth=2)
    ax2.set_title('Perplexity Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Learning rate
    ax3 = plt.subplot(n_rows, n_cols, 3)
    if 'learning_rate' in history:
        ax3.plot(history['learning_rate'], linewidth=2, color='orange')
        ax3.set_title('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Attention heatmap (if provided)
    if attention_data:
        ax4 = plt.subplot(n_rows, n_cols, (4, 6))  # Span 2 columns
        attention_weights = attention_data['weights']
        im = ax4.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax4.set_title('Sample Attention Matrix')
        
        if 'src_tokens' in attention_data:
            ax4.set_xticks(range(len(attention_data['src_tokens'])))
            ax4.set_xticklabels(attention_data['src_tokens'], rotation=45, ha='right')
        if 'tgt_tokens' in attention_data:
            ax4.set_yticks(range(len(attention_data['tgt_tokens'])))
            ax4.set_yticklabels(attention_data['tgt_tokens'])
        
        plt.colorbar(im, ax=ax4, shrink=0.6)
    
    # Training summary
    ax7 = plt.subplot(n_rows, n_cols, 7)
    summary_text = f"""
    Training Summary:
    
    Total Epochs: {len(history.get('train_loss', []))}
    Final Train Loss: {history.get('train_loss', [0])[-1]:.4f}
    """
    if 'val_loss' in history and history['val_loss']:
        summary_text += f"Best Val Loss: {min(history['val_loss']):.4f}\n"
    if 'train_perplexity' in history:
        summary_text += f"Final Train PPL: {history['train_perplexity'][-1]:.2f}\n"
    
    ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax7.set_title('Summary')
    ax7.axis('off')
    
    # Generation examples (if provided)
    if examples:
        ax8 = plt.subplot(n_rows, n_cols, (8, 9))  # Span 2 columns
        
        y_positions = []
        for i, example in enumerate(examples[:3]):  # Show top 3 examples
            y_pos = 3 - i
            y_positions.append(y_pos)
            
            # Input
            input_text = example.get('input', 'N/A')[:50] + '...' if len(example.get('input', '')) > 50 else example.get('input', 'N/A')
            ax8.text(0.05, y_pos + 0.3, f"In: {input_text}", fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
            
            # Generated
            generated_text = example.get('generated', 'N/A')[:50] + '...' if len(example.get('generated', '')) > 50 else example.get('generated', 'N/A')
            ax8.text(0.05, y_pos, f"Out: {generated_text}", fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
        
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 4)
        ax8.set_title('Generation Examples')
        ax8.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training dashboard saved to {save_path}")
    
    plt.show()

# Utility functions
def save_plot(fig, save_path: str, dpi: int = 300) -> None:
    """Save plot with high quality"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved to {save_path}")

def set_plot_style(style: str = 'seaborn', palette: str = 'husl') -> None:
    """Set plotting style and color palette"""
    plt.style.use(style)
    sns.set_palette(palette)

# Example usage
if __name__ == "__main__":
    print("Plotting utilities loaded successfully!")
    print("Available functions:")
    print("- plot_training_curves: Training progress visualization")
    print("- plot_attention_matrix: Attention heatmap visualization")
    print("- plot_model_comparison: Compare multiple models")
    print("- plot_generation_examples: Show generation examples")
    print("- plot_embeddings_2d: 2D embedding visualization")
    print("- create_training_dashboard: Comprehensive dashboard")