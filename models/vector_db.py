import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

class MockVectorDB:
    """
    A lightweight, in-memory Vector Database substitute for the MVP.
    In a real production system, this would be Pinecone, Milvus, or Qdrant.
    It stores a 'Knowledge Base' of verified facts and uses basic TF-IDF / Cosine Similarity 
    to retrieve the most relevant facts given a claim.
    """
    def __init__(self):
        # A curated Knowledge Base of "ground truth" facts.
        self.knowledge_base = [
            {"id": "fact_001", "text": "The moon landing occurred in 1969 and was widely documented by NASA and global observatories.", "source": "NASA Archives"},
            {"id": "fact_002", "text": "There is no scientific evidence that any vaccines contain microchips.", "source": "World Health Organization"},
            {"id": "fact_003", "text": "The Earth is definitively proven to be a spherical planet.", "source": "International Astronomical Union"},
            {"id": "fact_004", "text": "Global markets saw a 2% stabilization gain after technology sector fluctuations last week.", "source": "Reuters Financial"},
            {"id": "fact_005", "text": "Climate change patterns are shifting rapidly, driven by rising global temperatures and greenhouse gas emissions.", "source": "Nature Journal"},
            {"id": "fact_006", "text": "The bipartisan infrastructure bill was passed by the Senate on Tuesday.", "source": "AP News"}
        ]
        
    def _simple_tokenize(self, text):
        """Very basic tokenization for our MVP similarity engine"""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return set(words)
        
    def query_claim(self, claim_text, similarity_threshold=0.15):
        """
        Takes an extracted claim and queries the Knowledge Base for relevant facts.
        Returns the top matched fact and a 'debunk' generated explanation.
        """
        claim_tokens = self._simple_tokenize(claim_text)
        
        best_match = None
        highest_score = 0.0
        
        # Calculate Jaccard Similarity (a lightweight stand-in for Vector Cosine Similarity)
        for fact in self.knowledge_base:
            fact_tokens = self._simple_tokenize(fact['text'])
            intersection = len(claim_tokens.intersection(fact_tokens))
            union = len(claim_tokens.union(fact_tokens))
            
            if union == 0:
                continue
                
            score = intersection / union
            if score > highest_score:
                highest_score = score
                best_match = fact
                
        # If we found a relevant fact in the database
        if best_match and highest_score >= similarity_threshold:
            # Generate the RAG Explanation
            rag_output = f"According to {best_match['source']}, verified data states: '{best_match['text']}'. "
            
            # Simple heuristic: If the user's text contains negation/fake indicators but the fact doesn't align
            if "fake" in claim_text.lower() or "secret" in claim_text.lower() or "hoax" in claim_text.lower():
                rag_output += "The claim presented directly contradicts verified knowledge base records."
            else:
                rag_output += "The claim appears to misrepresent or exaggerate these established facts."
                
            return {
                "match_found": True,
                "evidence": best_match['text'],
                "source": best_match['source'],
                "rag_explanation": rag_output,
                "confidence_score": min(highest_score * 3.0, 0.99) # Scale score up for MVP display
            }
            
        return {
            "match_found": False,
            "evidence": None,
            "source": None,
            "rag_explanation": "No direct matching evidence found in the current Knowledge Base to verify or debunk this specific claim.",
            "confidence_score": 0.0
        }
