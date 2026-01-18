
import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

from models.bert_classifier import BERTClassifier
from app import extract_text_from_url

def test_model():
    print("--- Testing BERT Model ---")
    try:
        classifier = BERTClassifier()
        # Fake news example
        fake_text = "Breaking: Drinking bleach cures all diseases instantly! Doctors are hiding this secret."
        result_fake = classifier.predict(fake_text)
        print(f"Fake Text: '{fake_text}'")
        print(f"Prediction: {result_fake['prediction']} (Conf: {result_fake['confidence']:.2f})")

        # Real news example
        real_text = "NASA launches new rover to Mars to search for signs of ancient life."
        result_real = classifier.predict(real_text)
        print(f"Real Text: '{real_text}'")
        print(f"Prediction: {result_real['prediction']} (Conf: {result_real['confidence']:.2f})")
        
        return True
    except Exception as e:
        print(f"Model extraction failed: {e}")
        return False

def test_url():
    print("\n--- Testing URL Extraction ---")
    url = "https://www.bbc.com/news/world-us-canada-68937500" # Sample reliable URL (may not exist, trying generic)
    # Let's use a very stable URL
    url = "https://www.example.com"
    
    text = extract_text_from_url(url)
    print(f"URL: {url}")
    print(f"Extracted Text (first 100 chars): {text[:100]}...")
    
    if "Error" in text or "Could not" in text:
        print("URL Extraction Failed")
    else:
        print("URL Extraction Success")

if __name__ == "__main__":
    test_model()
    test_url()
