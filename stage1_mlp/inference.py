"""
Inference script for trained MLP models.

This script loads a trained MLP model and demonstrates how to use it
for text classification on new examples.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.mlp import SimpleMLP, MLPWithBagOfWords
from utils.data_utils import TextVocabulary


class MLPPredictor:
    """
    Wrapper class for making predictions with trained MLP models.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize predictor from saved model directory.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load vocabulary
        vocab_path = os.path.join(model_dir, 'vocabulary.pkl')
        self.vocabulary = TextVocabulary()
        self.vocabulary.load(vocab_path)
        
        # Initialize and load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✅ Model loaded from {model_dir}")
        print(f"Model type: {self.config['model_type']}")
        print(f"Classes: {self.config['class_names']}")
    
    def _load_model(self):
        """Load the trained model."""
        # Create model based on config
        if self.config['model_type'] == 'embedding':
            model = SimpleMLP(
                vocab_size=self.config['vocab_size'],
                embedding_dim=self.config['embedding_dim'],
                hidden_dims=self.config['hidden_dims'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout']
            )
            checkpoint_path = os.path.join(self.model_dir, 'best_embedding_mlp.pt')
        else:  # bag-of-words
            model = MLPWithBagOfWords(
                vocab_size=self.config['vocab_size'],
                hidden_dims=self.config['hidden_dims'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['dropout']
            )
            checkpoint_path = os.path.join(self.model_dir, 'best_bow_mlp.pt')
        
        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def predict(self, text: str) -> Tuple[int, float, List[float]]:
        """
        Predict the class of input text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        if self.config['model_type'] == 'embedding':
            return self._predict_embedding(text)
        else:
            return self._predict_bow(text)
    
    def _predict_embedding(self, text: str) -> Tuple[int, float, List[float]]:
        """Predict using embedding-based model."""
        # Preprocess text
        indices = self.vocabulary.text_to_indices(text, self.config['max_length'])
        
        # Create mask
        mask = [1 if i != self.vocabulary.token2idx[self.vocabulary.PAD_TOKEN] else 0 
                for i in indices]
        
        # Convert to tensors
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_tensor, mask_tensor)
            probabilities = F.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            all_probs = probabilities[0].cpu().numpy().tolist()
        
        return predicted_class, confidence, all_probs
    
    def _predict_bow(self, text: str) -> Tuple[int, float, List[float]]:
        """Predict using bag-of-words model."""
        # Create bag-of-words vector
        tokens = self.vocabulary.tokenize(text)
        vocab_size = len(self.vocabulary)
        bow_vector = torch.zeros(vocab_size, dtype=torch.float)
        
        for token in tokens:
            idx = self.vocabulary.token2idx.get(
                token, 
                self.vocabulary.token2idx[self.vocabulary.UNK_TOKEN]
            )
            bow_vector[idx] += 1.0
        
        # Convert to tensor
        input_tensor = bow_vector.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            all_probs = probabilities[0].cpu().numpy().tolist()
        
        return predicted_class, confidence, all_probs
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float, List[float]]]:
        """Predict classes for multiple texts."""
        return [self.predict(text) for text in texts]
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from class index."""
        return self.config['class_names'][class_idx]


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained MLP model')
    parser.add_argument('--model_dir', type=str, default='./checkpoints',
                       help='Directory containing trained model')
    parser.add_argument('--text', type=str,
                       help='Single text to classify')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        print("Please train a model first using train.py")
        return
    
    try:
        # Initialize predictor
        predictor = MLPPredictor(args.model_dir)
        
        if args.text:
            # Single prediction
            pred_class, confidence, probs = predictor.predict(args.text)
            class_name = predictor.get_class_name(pred_class)
            
            print(f"\n📝 Input text: {args.text}")
            print(f"🎯 Predicted class: {class_name} (index: {pred_class})")
            print(f"🎯 Confidence: {confidence:.4f}")
            print(f"\n📊 All probabilities:")
            for i, (prob, name) in enumerate(zip(probs, predictor.config['class_names'])):
                print(f"  {name}: {prob:.4f}")
        
        elif args.interactive:
            # Interactive mode
            print("\n🤖 Interactive MLP Classifier")
            print("Type 'quit' to exit")
            print("-" * 50)
            
            while True:
                try:
                    text = input("\nEnter text to classify: ").strip()
                    
                    if text.lower() == 'quit':
                        break
                    
                    if not text:
                        print("Please enter some text.")
                        continue
                    
                    pred_class, confidence, probs = predictor.predict(text)
                    class_name = predictor.get_class_name(pred_class)
                    
                    print(f"\n🎯 Prediction: {class_name} ({confidence:.4f})")
                    
                    # Show top 2 predictions
                    sorted_results = sorted(
                        enumerate(probs), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    print("📊 Top predictions:")
                    for i, (class_idx, prob) in enumerate(sorted_results[:2]):
                        name = predictor.get_class_name(class_idx)
                        print(f"  {i+1}. {name}: {prob:.4f}")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            print("\n👋 Goodbye!")
        
        else:
            # Demo with sample texts
            sample_texts = [
                "Stock market reaches new record high today",
                "Scientists discover breakthrough in cancer research",
                "Football team wins championship after amazing season",
                "Government announces new environmental policies"
            ]
            
            print("\n🧪 Demo predictions:")
            print("=" * 60)
            
            for i, text in enumerate(sample_texts, 1):
                pred_class, confidence, probs = predictor.predict(text)
                class_name = predictor.get_class_name(pred_class)
                
                print(f"\n{i}. Text: {text}")
                print(f"   Prediction: {class_name} ({confidence:.4f})")
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please make sure you have trained a model first.")


if __name__ == '__main__':
    main()
