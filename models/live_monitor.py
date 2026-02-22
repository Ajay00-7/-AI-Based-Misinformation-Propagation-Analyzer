import random
from datetime import datetime

class LiveMonitor:
    """
    Mock implementation of a Real-Time RSS / API News Monitor.
    In production, this connects to Twitter API, Reddit API, and News RSS feeds.
    """
    def __init__(self):
        # A pool of headlines to simulate a live incoming data stream
        self.mock_headlines = [
            {
                "source": "Twitter/X (Flagged Account)",
                "headline": "BREAKING: Secret documents prove the moon landing was filmed in a Hollywood basement!",
                "url": "https://twitter.com/conspiracy_truth/status/123",
                "timestamp": None
            },
            {
                "source": "Reuters RSS",
                "headline": "Global markets stabilize after recent technology sector fluctuations",
                "url": "https://reuters.com/markets/123",
                "timestamp": None
            },
            {
                "source": "Reddit r/news",
                "headline": "Scientists discover new species of deep sea coral in the Pacific Ocean",
                "url": "https://reddit.com/r/news/comments/123",
                "timestamp": None
            },
            {
                "source": "Unknown Blog",
                "headline": "100% PROOF that the government is poisoning the water supply to control minds!!!",
                "url": "http://truth-seeker-blog.net/water",
                "timestamp": None
            },
            {
                "source": "AP News",
                "headline": "Senate passes new infrastructure bill with bipartisan support",
                "url": "https://apnews.com/article/123",
                "timestamp": None
            }
        ]
        
    def fetch_latest_items(self, count=3):
        """
        Simulates fetching the latest news items across monitored platforms.
        """
        items = random.sample(self.mock_headlines, min(count, len(self.mock_headlines)))
        
        # Add current timestamps
        for i, item in enumerate(items):
            # Stagger the timestamps slightly to look like a live feed
            minutes_ago = random.randint(1, 15)
            item['timestamp'] = f"{minutes_ago} mins ago"
            
        return items
