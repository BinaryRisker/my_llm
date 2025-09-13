"""
Universal Training Utilities
============================

This module provides common training utilities that can be used across
all stages of the project, including training loops, optimization,
checkpointing, and monitoring.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import os
import json
import logging
from typing import Dict, Any, Optional, Callable, Union, List
from pathlib import Path

class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self,
                 model_name: str = "model",
                 num_epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 gradient_clip_norm: Optional[float] = 1.0,
                 warmup_steps: int = 0,
                 save_every: int = 5,
                 eval_every: int = 1,
                 early_stopping_patience: int = 10,
                 checkpoint_dir: str = "checkpoints",
                 log_level: str = "INFO",
                 device: str = "auto"):
        
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.warmup_steps = warmup_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_level = log_level
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

class UniversalTrainer:
    """Universal trainer that works with any PyTorch model"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 criterion: Optional[nn.Module] = None):
        
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(), 
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.criterion = criterion
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': []
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.checkpoint_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_fn: Custom training function, if None uses default
            
        Returns:
            Dictionary with training metrics
        """
        if train_fn is None:
            train_fn = self._default_train_step
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.config.device) if torch.is_tensor(b) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.config.device)
            
            # Forward pass
            loss = train_fn(self.model, batch, self.criterion)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step (for step-based schedulers)
            if self.scheduler is not None and hasattr(self.scheduler, 'step'):
                if hasattr(self.scheduler, 'step_batch'):
                    self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Warmup learning rate
            if self.global_step <= self.config.warmup_steps:
                lr = self.config.learning_rate * self.global_step / self.config.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, "
                    f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        
        epoch_time = time.time() - start_time
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'time': epoch_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_fn: Custom validation function, if None uses default
            
        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        if val_fn is None:
            val_fn = self._default_val_step
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.config.device) if torch.is_tensor(b) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.config.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.config.device)
                
                # Forward pass
                loss = val_fn(self.model, batch, self.criterion)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 10))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def train(self, 
              train_fn: Optional[Callable] = None,
              val_fn: Optional[Callable] = None,
              callback_fn: Optional[Callable] = None) -> Dict[str, List]:
        """
        Full training loop
        
        Args:
            train_fn: Custom training function
            val_fn: Custom validation function  
            callback_fn: Callback function called after each epoch
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_fn)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_perplexity'].append(train_metrics['perplexity'])
            self.training_history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Validate
            val_metrics = {}
            if self.val_dataloader is not None and epoch % self.config.eval_every == 0:
                val_metrics = self.validate(val_fn)
                if val_metrics:
                    self.training_history['val_loss'].append(val_metrics['loss'])
                    self.training_history['val_perplexity'].append(val_metrics['perplexity'])
            
            # Log epoch results
            log_msg = f"Epoch {epoch+1}/{self.config.num_epochs} - "
            log_msg += f"Train Loss: {train_metrics['loss']:.4f}, "
            log_msg += f"Train PPL: {train_metrics['perplexity']:.2f}"
            
            if val_metrics:
                log_msg += f", Val Loss: {val_metrics['loss']:.4f}"
                log_msg += f", Val PPL: {val_metrics['perplexity']:.2f}"
            
            log_msg += f", Time: {train_metrics['time']:.1f}s"
            self.logger.info(log_msg)
            
            # Scheduler step (for epoch-based schedulers)
            if self.scheduler is not None and not hasattr(self.scheduler, 'step_batch'):
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
            
            # Early stopping
            if val_metrics:
                val_loss = val_metrics['loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self.save_checkpoint("best_model.pth")
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Callback
            if callback_fn is not None:
                callback_fn(self, epoch, train_metrics, val_metrics)
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        self.save_training_history()
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.config.checkpoint_dir / filename)
        self.logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'val_loss': [], 'train_perplexity': [], 
            'val_perplexity': [], 'learning_rate': []
        })
        
        self.logger.info(f"Checkpoint loaded: {filename}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.config.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")
    
    def _default_train_step(self, model, batch, criterion):
        """Default training step for sequence-to-sequence tasks"""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            src, tgt = batch[0], batch[1]
            
            # For Transformer-like models
            if hasattr(model, 'forward') and len(batch) >= 2:
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = model(src, tgt_input)
                if isinstance(logits, dict):
                    logits = logits['logits']
                
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            else:
                # Fallback
                output = model(src)
                loss = criterion(output, tgt)
        else:
            # Single input case
            output = model(batch)
            # Assume self-supervised learning task
            loss = criterion(output[:, :-1].reshape(-1, output.size(-1)), 
                           batch[:, 1:].reshape(-1))
        
        return loss
    
    def _default_val_step(self, model, batch, criterion):
        """Default validation step"""
        return self._default_train_step(model, batch, criterion)

def create_optimizer(model: nn.Module, 
                    optimizer_type: str = "adamw",
                    learning_rate: float = 0.001,
                    weight_decay: float = 0.01,
                    **kwargs) -> optim.Optimizer:
    """
    Create optimizer with common configurations
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, 
                          weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, 
                         weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, 
                        weight_decay=weight_decay, momentum=kwargs.get('momentum', 0.9))
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_type: str = "cosine",
                    num_epochs: int = 100,
                    warmup_steps: int = 0,
                    **kwargs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler
    """
    if scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, **kwargs
        )
    elif scheduler_type.lower() == 'linear':
        return optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 30), 
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=kwargs.get('patience', 10)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

# Example usage
if __name__ == "__main__":
    # This would be used with actual models and data
    print("Training utilities loaded successfully!")
    print("Available components:")
    print("- TrainingConfig: Configuration management")
    print("- UniversalTrainer: General training loop")
    print("- create_optimizer: Optimizer factory")
    print("- create_scheduler: Scheduler factory")