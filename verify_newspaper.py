
import nltk
from newspaper import Article

def verify_extraction():
    url = "https://www.bbc.com/news/world-us-canada-68937500" # Use a stable URL or generic one
    url = "https://www.google.com" # Simplest check for connectivity and library load
    
    print(f"Testing newspaper3k on {url}...")
    try:
        article = Article(url)
        article.download()
        article.parse()
        print("Success! Title:", article.title)
        print("Text snippet:", article.text[:50])
    except Exception as e:
        print("Error:", e)
        # Check for NLTK error
        if "punkt" in str(e):
            print("Downloading missing NLTK data...")
            nltk.download('punkt')
            print("Retrying...")
            article.download()
            article.parse()
            print("Success after download! Title:", article.title)

if __name__ == "__main__":
    # Pre-download punkt to be safe, it's small
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    verify_extraction()
