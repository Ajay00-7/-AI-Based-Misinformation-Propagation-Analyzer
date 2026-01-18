
import sys
import os
import time

# Add current directory to path
sys.path.append(os.getcwd())

print("Importing app modules...")
try:
    from models.bert_classifier import BERTClassifier
    print("BERTClassifier imported.")
except ImportError as e:
    print(f"Error importing BERTClassifier: {e}")

try:
    from models.svm_baseline import SVMBaseline
    print("SVMBaseline imported.")
except ImportError as e:
    print(f"Error importing SVMBaseline: {e}")

def debug_run():
    print("Starting debug run...")
    text = "This is a test article."
    
    print("Initializing BERT model...")
    bert = BERTClassifier()
    print("Loading BERT weights (this might download)...")
    start = time.time()
    bert.load_model()
    print(f"BERT loaded in {time.time() - start:.2f}s")
    
    print("Running BERT prediction...")
    res = bert.predict(text)
    print("BERT Result:", res)
    
    print("Initializing SVM model...")
    svm = SVMBaseline()
    # svm.train... might trigger training
    
    print("Debug run complete.")

if __name__ == "__main__":
    debug_run()
