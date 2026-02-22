import re

class ClaimExtractor:
    """
    NLP Module to extract potentially verifiable claims from text.
    In a production RAG system, these claims would be queried against
    a Vector DB (e.g., Pinecone) of verified facts.
    """
    def __init__(self):
        # Heuristic rules for claim extraction:
        # Looking for sentences with numbers, strong descriptive verbs, or proper nouns
        self.claim_indicators = [
            r'\b(?:is|are|was|were|will|has|have|had)\b',
            r'\b(?:percent|%|dollars|\$|million|billion|thousand)\b',
            r'\d+'
        ]
        
    def extract_claims(self, text):
        """
        Extracts specific claims from a block of text.
        """
        # Simple sentence tokenization based on punctuation
        sentences = re.split(r'(?<=[.!?]) +', text.replace('\n', ' '))
        extracted_claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 4 or len(sentence.split()) > 40:
                continue # Skip very small fragments or massive run-on lines
            
            # Score the sentence based on heuristics
            score = 0
            for indicator in self.claim_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    score += 1
                    
            if score >= 2: # At least 2 indicators to be considered a strong, verifiable claim
                extracted_claims.append(sentence)
                
        return {
            "total_claims_found": len(extracted_claims),
            "claims": extracted_claims[:5] # Return top 5 claims to prevent UI overflow
        }
