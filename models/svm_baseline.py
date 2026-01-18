"""
SVM Baseline Classifier for Misinformation Detection
Uses TF-IDF + Support Vector Machine for comparison with BERT
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pickle
import os
import re

class SVMBaseline:
    def __init__(self, max_features=5000):
        """
        Initialize SVM baseline classifier
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english'
        )
        self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        self.is_trained = False
        
    def train_on_dataset(self, dataset_path):
        """
        Train SVM classifier on dataset
        
        Args:
            dataset_path: Path to CSV file with 'text' and 'label' columns
            
        Returns:
            dict: Training metrics
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize
        print("Vectorizing text with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train SVM
        print(f"Training SVM on {len(X_train)} samples...")
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed! Test Accuracy: {accuracy:.2%}")
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict(self, text):
        """
        Predict if news is fake or real
        
        Args:
            text: News article text
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            # Train on default dataset if not trained
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
            if os.path.exists(dataset_path):
                self.train_on_dataset(dataset_path)
            else:
                raise ValueError("Model not trained and dataset not found!")
        
        # Vectorize input
        text_tfidf = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.classifier.predict(text_tfidf)[0]
        probabilities = self.classifier.predict_proba(text_tfidf)[0]
        
        # Map to labels
        label_map = {0: 'Real', 1: 'Fake'}
        prediction_label = label_map[prediction]
        confidence = float(probabilities[prediction])
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': {
                'Real': float(probabilities[0]),
                'Fake': float(probabilities[1])
            },
            'label_code': int(prediction)
        }
    
    def evaluate_on_dataset(self, dataset_path):
        """
        Evaluate model on dataset
        
        Args:
            dataset_path: Path to CSV file
            
        Returns:
            dict: Accuracy metrics
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
    
    def get_top_features(self, n=20):
        """
        Get top features for each class
        
        Args:
            n: Number of top features to return
            
        Returns:
            dict: Top features for Real and Fake classes
        """
        if not self.is_trained:
            return None
        
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_[0]
        
        # Top features for Real (negative coefficients)
        real_indices = np.argsort(coef)[:n]
        real_features = [(feature_names[i], coef[i]) for i in real_indices]
        
        # Top features for Fake (positive coefficients)
        fake_indices = np.argsort(coef)[-n:][::-1]
        fake_features = [(feature_names[i], coef[i]) for i in fake_indices]
        
        return {
            'real_indicators': real_features,
            'fake_indicators': fake_features
        }

    def explain_prediction(self, text, top_n=5):
        """
        Explain why a prediction was made by showing top contributing words
        
        Args:
            text: Input text
            top_n: Number of words to highlight
            
        Returns:
            dict: {
                'contributing_words': list of (word, score),
                'class': 'Fake' or 'Real'
            }
        """
        if not self.is_trained:
            return {'contributing_words': [], 'class': 'Unknown'}
        
        # Transform input
        tfidf_vector = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get separate words from input (simple split for mapping)
        input_words = set(re.findall(r'\w+', text.lower()))
        
        # Get nonzero elements (indices of words present in this text)
        coo = tfidf_vector.tocoo()
        word_indices = coo.col
        
        # Get coefficients
        if hasattr(self.classifier.coef_, 'toarray'):
            coef = self.classifier.coef_.toarray()[0]
        else:
            coef = self.classifier.coef_[0]
        
        # Calculate contribution: tfidf_value * coefficient
        contributions = []
        for idx in word_indices:
            word = feature_names[idx]
            # only care if word is in our input text vocabulary check
            if word in input_words or any(w in word for w in input_words):
                 # Handle sparse or dense data access
                 if hasattr(coo.data, 'toarray'): # unlikely for coo.data but just in case
                     val = coo.data[np.where(coo.col == idx)[0][0]]
                 else:
                     val = coo.data[np.where(coo.col == idx)[0][0]]
                     
                 score = coef[idx] * val
                 contributions.append((word, score))
        
        # Sort contributions
        # Positive score -> Fake, Negative score -> Real
        contributions.sort(key=lambda x: x[1])
        
        prediction = self.predict(text)['prediction']
        
        if prediction == 'Fake':
            # return top positive scores (descending)
            top_words = sorted([c for c in contributions if c[1] > 0], key=lambda x: x[1], reverse=True)[:top_n]
        else:
            # return top negative scores (ascending)
            top_words = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])[:top_n]
            
        return {
            'contributing_words': [{"word": w, "score": float(s)} for w, s in top_words],
            'prediction': prediction
        }


# For quick testing
if __name__ == "__main__":
    svm = SVMBaseline()
    
    # Train on dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    if os.path.exists(dataset_path):
        metrics = svm.train_on_dataset(dataset_path)
        print(f"\nAccuracy: {metrics['accuracy']:.2%}")
        
        # Test predictions
        sample_fake = "Breaking: Drinking bleach cures all diseases!"
        sample_real = "Scientists publish new research in medical journal."
        
        print("\nFake News Test:")
        result1 = svm.predict(sample_fake)
        print(f"Prediction: {result1['prediction']} (Confidence: {result1['confidence']:.2%})")
        
        print("\nReal News Test:")
        result2 = svm.predict(sample_real)
        print(f"Prediction: {result2['prediction']} (Confidence: {result2['confidence']:.2%})")
        
        # Show top features
        print("\nTop Features:")
        features = svm.get_top_features(10)
        print("Fake Indicators:", [f[0] for f in features['fake_indicators'][:5]])
        print("Real Indicators:", [f[0] for f in features['real_indicators'][:5]])
