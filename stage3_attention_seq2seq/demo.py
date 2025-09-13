#!/usr/bin/env python3
"""
Stage 3 Attention & Seq2Seq - Interactive Demo
==============================================

This script provides an interactive demonstration of the attention mechanisms
and sequence-to-sequence models implemented in Stage 3.

Features:
- Quick model training with sample data
- Interactive translation with attention visualization
- BLEU score evaluation
- Attention pattern analysis

Author: AI Assistant
Date: 2024
"""

import os
import sys
import torch
import json
import random
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.seq2seq import Seq2SeqModel
from models.attention import create_attention
from models.encoder import create_encoder
from models.decoder import AttentionDecoder
from utils.data_loader import TranslationDataset, create_vocab
from utils.training import train_model, translate_sentence
from visualize import AttentionVisualizer

def create_sample_data():
    """Create sample English-French translation pairs for demo"""
    sample_pairs = [
        ("hello", "bonjour"),
        ("world", "monde"),
        ("cat", "chat"),
        ("dog", "chien"),
        ("house", "maison"),
        ("car", "voiture"),
        ("book", "livre"),
        ("water", "eau"),
        ("food", "nourriture"),
        ("good", "bon"),
        ("morning", "matin"),
        ("evening", "soir"),
        ("night", "nuit"),
        ("today", "aujourd hui"),
        ("hello world", "bonjour monde"),
        ("i love cats", "j aime les chats"),
        ("the book is good", "le livre est bon"),
        ("we are happy", "nous sommes heureux"),
        ("good morning", "bonjour"),
        ("how are you", "comment allez vous"),
        ("thank you", "merci"),
        ("see you later", "a bientot"),
        ("i want water", "je veux de l eau"),
        ("the cat is sleeping", "le chat dort"),
        ("this is a good book", "c est un bon livre"),
        ("we love our house", "nous aimons notre maison"),
        ("the car is fast", "la voiture est rapide"),
        ("good evening", "bonsoir"),
        ("have a good day", "bonne journee"),
        ("where is the book", "ou est le livre")
    ]
    
    return sample_pairs

def quick_train_demo_model(data_pairs, attention_type='bahdanau', hidden_size=128, 
                          num_epochs=10, device=None):
    """
    Train a small demo model quickly for demonstration purposes
    
    Args:
        data_pairs: List of (source, target) pairs
        attention_type: Type of attention mechanism
        hidden_size: Hidden dimension size
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Trained model and vocabulary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Training demo model with {attention_type} attention...")
    print(f"📊 Data: {len(data_pairs)} pairs")
    print(f"💻 Device: {device}")
    print(f"🏗️ Hidden size: {hidden_size}, Epochs: {num_epochs}")
    
    # Create vocabulary
    src_sentences = [pair[0] for pair in data_pairs]
    tgt_sentences = [pair[1] for pair in data_pairs]
    
    src_vocab = create_vocab(src_sentences, min_freq=1, max_size=1000)
    tgt_vocab = create_vocab(tgt_sentences, min_freq=1, max_size=1000)
    
    print(f"📚 Vocabulary: {len(src_vocab)} source words, {len(tgt_vocab)} target words")
    
    # Create dataset
    dataset = TranslationDataset(data_pairs, src_vocab, tgt_vocab)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, 
                                            collate_fn=dataset.collate_fn)
    
    # Create model components
    encoder = create_encoder(
        vocab_size=len(src_vocab),
        embed_size=hidden_size//2,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1,
        encoder_type='lstm',
        bidirectional=True
    )
    
    attention = create_attention(
        attention_type=attention_type,
        hidden_size=hidden_size,
        attention_dim=hidden_size//2
    )
    
    decoder = AttentionDecoder(
        vocab_size=len(tgt_vocab),
        embed_size=hidden_size//2,
        hidden_size=hidden_size,
        attention=attention,
        num_layers=1,
        dropout=0.1
    )
    
    # Create full model
    model = Seq2SeqModel(encoder, decoder).to(device)
    
    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Quick training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            src, tgt, src_mask, tgt_mask = batch
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with teacher forcing
            tgt_input = tgt[:, :-1]  # All but last token
            tgt_output = tgt[:, 1:]  # All but first token
            
            output, _ = model(src, tgt_input, src_mask, tgt_mask[:, :-1])
            
            # Compute loss
            loss = criterion(output.reshape(-1, output.size(-1)), 
                           tgt_output.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"📈 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("✅ Training completed!")
    
    return model, src_vocab, tgt_vocab

def interactive_demo(model, src_vocab, tgt_vocab, device):
    """
    Run interactive translation demo
    
    Args:
        model: Trained model
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device model is on
    """
    print("\n🌐 Interactive Translation Demo")
    print("=" * 50)
    print("Enter English sentences to translate to French")
    print("Commands: 'quit' to exit, 'examples' for sample sentences")
    
    # Create vocabulary mappings
    src_word2id = {word: i for i, word in enumerate(src_vocab)}
    tgt_word2id = {word: i for i, word in enumerate(tgt_vocab)}
    tgt_id2word = {i: word for word, i in tgt_word2id.items()}
    
    sample_sentences = [
        "hello world",
        "good morning", 
        "i love cats",
        "the book is good",
        "thank you"
    ]
    
    while True:
        user_input = input("\n📝 Enter sentence (or command): ").strip().lower()
        
        if user_input in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input in ['examples', 'help']:
            print("📖 Sample sentences to try:")
            for i, sent in enumerate(sample_sentences, 1):
                print(f"   {i}. {sent}")
            continue
        
        if not user_input:
            continue
        
        try:
            # Tokenize and convert to indices
            tokens = user_input.split()
            indices = [src_word2id.get(token, src_word2id.get('<UNK>', 3)) for token in tokens]
            indices = [src_word2id.get('<SOS>', 1)] + indices + [src_word2id.get('<EOS>', 2)]
            
            # Translate
            src_tensor = torch.tensor(indices).unsqueeze(0).to(device)
            src_mask = torch.ones_like(src_tensor, dtype=torch.bool)
            
            model.eval()
            with torch.no_grad():
                # Encode
                encoder_outputs, encoder_hidden = model.encoder(src_tensor, src_mask)
                
                # Decode
                decoder_input = torch.tensor([[tgt_word2id.get('<SOS>', 1)]]).to(device)
                decoder_hidden = encoder_hidden
                
                translated_words = []
                attention_weights = []
                
                for _ in range(20):  # Max length
                    decoder_output, decoder_hidden, attention = model.decoder(
                        decoder_input, decoder_hidden, encoder_outputs, src_mask
                    )
                    
                    next_token = decoder_output.argmax(dim=-1)
                    token_id = next_token.item()
                    
                    if token_id == tgt_word2id.get('<EOS>', 2):
                        break
                    
                    word = tgt_id2word.get(token_id, '<UNK>')
                    if word not in ['<PAD>', '<SOS>']:
                        translated_words.append(word)
                    
                    if attention is not None:
                        attention_weights.append(attention.squeeze(0).cpu().numpy())
                    
                    decoder_input = next_token.unsqueeze(0)
            
            translation = ' '.join(translated_words)
            print(f"🔤 Input: {user_input}")
            print(f"🔄 Translation: {translation}")
            
            # Show attention if available
            if attention_weights:
                show_attention = input("📊 Show attention visualization? (y/n): ").lower().startswith('y')
                if show_attention:
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        # Simple attention visualization
                        src_tokens = ['<SOS>'] + tokens + ['<EOS>']
                        attention_matrix = np.array(attention_weights[:len(translated_words)])
                        attention_matrix = attention_matrix[:, :len(src_tokens)]
                        
                        plt.figure(figsize=(10, 6))
                        plt.imshow(attention_matrix, cmap='Blues', aspect='auto')
                        plt.xlabel('Source Tokens')
                        plt.ylabel('Target Tokens')
                        plt.title(f'Attention Weights\n"{user_input}" → "{translation}"')
                        
                        plt.xticks(range(len(src_tokens)), src_tokens, rotation=45)
                        plt.yticks(range(len(translated_words)), translated_words)
                        
                        # Add text annotations
                        for i in range(len(translated_words)):
                            for j in range(min(len(src_tokens), attention_matrix.shape[1])):
                                plt.text(j, i, f'{attention_matrix[i, j]:.2f}',
                                       ha="center", va="center", fontsize=8,
                                       color="white" if attention_matrix[i, j] > 0.5 else "black")
                        
                        plt.colorbar(label='Attention Weight')
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        print(f"❌ Could not show visualization: {e}")
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            print("Try a simpler sentence or check vocabulary coverage")

def evaluate_demo_model(model, src_vocab, tgt_vocab, test_pairs, device):
    """
    Evaluate the demo model on test pairs
    
    Args:
        model: Trained model
        src_vocab: Source vocabulary  
        tgt_vocab: Target vocabulary
        test_pairs: List of (source, target) test pairs
        device: Device model is on
        
    Returns:
        Evaluation results
    """
    print(f"\n📊 Evaluating model on {len(test_pairs)} test pairs...")
    
    src_word2id = {word: i for i, word in enumerate(src_vocab)}
    tgt_word2id = {word: i for i, word in enumerate(tgt_vocab)}
    tgt_id2word = {i: word for word, i in tgt_word2id.items()}
    
    model.eval()
    correct = 0
    total = 0
    
    for source, target in test_pairs[:10]:  # Evaluate on first 10 pairs
        try:
            # Tokenize source
            src_tokens = source.split()
            src_indices = [src_word2id.get(token, src_word2id.get('<UNK>', 3)) for token in src_tokens]
            src_indices = [src_word2id.get('<SOS>', 1)] + src_indices + [src_word2id.get('<EOS>', 2)]
            
            # Translate
            src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
            src_mask = torch.ones_like(src_tensor, dtype=torch.bool)
            
            with torch.no_grad():
                encoder_outputs, encoder_hidden = model.encoder(src_tensor, src_mask)
                decoder_input = torch.tensor([[tgt_word2id.get('<SOS>', 1)]]).to(device)
                decoder_hidden = encoder_hidden
                
                translated_words = []
                for _ in range(15):
                    decoder_output, decoder_hidden, _ = model.decoder(
                        decoder_input, decoder_hidden, encoder_outputs, src_mask
                    )
                    
                    next_token = decoder_output.argmax(dim=-1)
                    token_id = next_token.item()
                    
                    if token_id == tgt_word2id.get('<EOS>', 2):
                        break
                    
                    word = tgt_id2word.get(token_id, '<UNK>')
                    if word not in ['<PAD>', '<SOS>']:
                        translated_words.append(word)
                    
                    decoder_input = next_token.unsqueeze(0)
            
            translation = ' '.join(translated_words)
            
            # Simple accuracy check (exact match)
            if translation.strip() == target.strip():
                correct += 1
            
            total += 1
            
            print(f"   Source: {source}")
            print(f"   Target: {target}")
            print(f"   Predicted: {translation}")
            print(f"   {'✅' if translation.strip() == target.strip() else '❌'}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error translating '{source}': {e}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"📈 Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    return accuracy

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Stage 3 Attention Seq2Seq Demo')
    parser.add_argument('--attention_type', type=str, default='bahdanau',
                       choices=['bahdanau', 'luong', 'coverage'],
                       help='Attention mechanism type')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and use existing model')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive mode only')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only')
    
    args = parser.parse_args()
    
    print("🎯 Stage 3 Attention & Seq2Seq Demo")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_data = create_sample_data()
    
    # Split data for train/test
    random.shuffle(sample_data)
    split_idx = int(0.8 * len(sample_data))
    train_data = sample_data[:split_idx]
    test_data = sample_data[split_idx:]
    
    model_path = f"demo_model_{args.attention_type}.pth"
    
    # Load or train model
    if args.skip_training and os.path.exists(model_path):
        print(f"📂 Loading existing model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = checkpoint['model']
        src_vocab = checkpoint['src_vocab']
        tgt_vocab = checkpoint['tgt_vocab']
    else:
        # Train new model
        model, src_vocab, tgt_vocab = quick_train_demo_model(
            train_data, 
            attention_type=args.attention_type,
            hidden_size=args.hidden_size,
            num_epochs=args.num_epochs,
            device=device
        )
        
        # Save model
        checkpoint = {
            'model': model,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'attention_type': args.attention_type
        }
        torch.save(checkpoint, model_path)
        print(f"💾 Model saved to {model_path}")
    
    # Run selected mode
    if args.interactive:
        interactive_demo(model, src_vocab, tgt_vocab, device)
    elif args.evaluate:
        evaluate_demo_model(model, src_vocab, tgt_vocab, test_data, device)
    else:
        # Run full demo
        print("\n🧪 Running evaluation...")
        accuracy = evaluate_demo_model(model, src_vocab, tgt_vocab, test_data, device)
        
        print(f"\n🎉 Demo completed!")
        print(f"📊 Final accuracy: {accuracy:.2%}")
        
        run_interactive = input("\n🤔 Run interactive demo? (y/n): ").lower().startswith('y')
        if run_interactive:
            interactive_demo(model, src_vocab, tgt_vocab, device)
    
    print("\n✨ Thank you for trying the Stage 3 demo!")
    print("Next: Train on larger datasets or try different attention mechanisms")

if __name__ == "__main__":
    main()