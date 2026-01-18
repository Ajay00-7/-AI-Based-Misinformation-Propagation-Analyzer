"""
BERT-based Misinformation Classifier
Uses HuggingFace Transformers for binary classification (Real/Fake)
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class BERTClassifier:
    def __init__(self, model_name='mrm8488/bert-tiny-finetuned-fake-news-detection', max_length=512):
        """
        Initialize BERT classifier
        
        Args:
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Load or initialize model
        self.model = None
        self.is_trained = True # Pre-trained model
        
    def load_model(self):
        """Load pre-trained BERT model for sequence classification"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2  # Binary: 0=Fake, 1=Real (Check specific model config if needed, usually 0/1)
        )
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_text(self, text):
        """
        Preprocess and tokenize text
        
        Args:
            text: Input text string
            
        Returns:
            Tokenized input tensors
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, text):
        """
        Predict if news is fake or real
        
        Args:
            text: News article text
            
        Returns:
            dict: {
                'prediction': 'Fake' or 'Real',
                'confidence': float (0-1),
                'probabilities': {'Real': float, 'Fake': float}
            }
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess
        inputs = self.preprocess_text(text)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
        # Get prediction
        probs = probabilities.cpu().numpy()[0]
        prediction_idx = np.argmax(probs)
        confidence = float(probs[prediction_idx])
        
        # Map to labels
        # For mrm8488 model: Check specific mapping. Usually 1=Fake, 0=Real
        label_map = {0: 'Real', 1: 'Fake'}
        prediction = label_map.get(prediction_idx, "Unknown")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'Real': float(probs[0]),
                'Fake': float(probs[1])
            },
            'label_code': int(prediction_idx)
        }
    
    def train_on_dataset(self, dataset_path, epochs=3, batch_size=8, learning_rate=2e-5):
        """
        Fine-tune BERT on custom dataset
        
        Args:
            dataset_path: Path to CSV file with 'text' and 'label' columns
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Tokenize
        train_encodings = [self.preprocess_text(text) for text in train_texts]
        val_encodings = [self.preprocess_text(text) for text in val_texts]
        
        # Create datasets
        train_input_ids = torch.cat([enc['input_ids'] for enc in train_encodings])
        train_attention_masks = torch.cat([enc['attention_mask'] for enc in train_encodings])
        train_labels_tensor = torch.tensor(train_labels)
        
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model if not loaded
        if self.model is None:
            self.load_model()
        
        # Set to training mode
        self.model.train()
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"Training BERT on {len(train_texts)} samples...")
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids, attention_mask, labels = batch
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self.is_trained = True
        print("Training completed!")
        
        return self.evaluate_on_dataset(dataset_path)
    
    def evaluate_on_dataset(self, dataset_path):
        """
        Evaluate model on dataset
        
        Args:
            dataset_path: Path to CSV file
            
        Returns:
            dict: Accuracy and other metrics
        """
        df = pd.read_csv(dataset_path)
        
        correct = 0
        total = len(df)
        
        for _, row in df.iterrows():
            result = self.predict(row['text'])
            predicted_label = result['label_code']
            if predicted_label == row['label']:
                correct += 1
        
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


# For quick testing
if __name__ == "__main__":
    classifier = BERTClassifier()
    
    # Test with sample text
    sample_fake = "Breaking: Drinking bleach cures all diseases!"
    sample_real = "Scientists publish new research in medical journal."
    
    print("Testing BERT Classifier...")
    print("\nFake News Test:")
    result1 = classifier.predict(sample_fake)
    print(f"Prediction: {result1['prediction']} (Confidence: {result1['confidence']:.2%})")
    
    print("\nReal News Test:")
    result2 = classifier.predict(sample_real)
    print(f"Prediction: {result2['prediction']} (Confidence: {result2['confidence']:.2%})")
