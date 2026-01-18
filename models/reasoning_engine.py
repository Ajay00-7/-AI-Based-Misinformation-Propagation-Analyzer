"""
Reasoning Engine for Misinformation Analysis
Generates natural language explanations ("Executive Summaries") based on linguistic features.
Acts as a logical layer on top of the black-box BERT/SVM models.
"""

import re
import random

class ReasoningEngine:
    def __init__(self):
        # Emotional trigger words often found in misinformation
        self.emotional_triggers = [
            'shocking', 'banned', 'exposed', 'secret', 'panic', 'death', 'illegal',
            'miracle', 'worst', 'best', 'you won\'t believe', 'conspiracy', 'cover-up'
        ]
        
        # Absolute terms that lack nuance
        self.absolutisms = [
            'always', 'never', 'everyone', 'nobody', 'absolutely', 'undeniable', 'proof'
        ]

    def analyze(self, text, prediction, confidence):
        """
        Generate a qualitative report based on the input text and model verdict.
        """
        analysis = {
            'emotional_intensity': self._calculate_emotional_intensity(text),
            'structural_issues': self._check_structure(text),
            'rhetorical_tactics': self._identify_tactics(text),
            'executive_summary': ''
        }
        
        analysis['executive_summary'] = self._generate_summary(analysis, prediction, confidence)
        return analysis

    def _calculate_emotional_intensity(self, text):
        matches = [word for word in self.emotional_triggers if word in text.lower()]
        score = len(matches)
        return {
            'score': score,
            'level': 'High' if score > 2 else 'Moderate' if score > 0 else 'Low',
            'triggers_found': list(set(matches))
        }

    def _check_structure(self, text):
        issues = []
        if text.isupper():
            issues.append("Excessive Capitalization")
        if text.count('!') > 3:
            issues.append("Excessive Exclamation Marks")
        if len(text.split()) < 50:
            issues.append("Very Short Content (Lacks Detail)")
        return issues

    def _identify_tactics(self, text):
        tactics = []
        lower_text = text.lower()
        
        if any(word in lower_text for word in self.absolutisms):
            tactics.append("Absolutism (Lack of Nuance)")
            
        if "propaganda" in lower_text or "brainwash" in lower_text:
            tactics.append("Ad Hominem / Attack Language")
            
        if "?" in text and ("fake" in lower_text or "truth" in lower_text):
            tactics.append("Betteridge's Law (Sensationalist Questioning)")
            
        return tactics

    def _generate_summary(self, analysis, verdict, confidence):
        """Construct a professional 'Consultant' style narrative"""
        summary = []
        
        # Opening Statement
        if verdict == "Fake":
            opener = f"Our AI Council has flagged this content as High Risk (Confidence: {confidence*100:.1f}%)."
        else:
            opener = f"Our AI Council has verified this content as likely Authentic (Confidence: {confidence*100:.1f}%)."
        summary.append(opener)
        
        # Evidence Layer
        if verdict == "Fake":
            if analysis['emotional_intensity']['level'] == 'High':
                words = ", ".join(analysis['emotional_intensity']['triggers_found'][:3])
                summary.append(f"Primary Scan detects high emotional manipulation, utilizing triggers such as '{words}'.")
            
            if analysis['structural_issues']:
                issues = ", ".join(analysis['structural_issues'])
                summary.append(f"Structural integrity is compromised: {issues}.")
                
            summary.append("The text prioritizes improved virality over informational density.")
        else:
            summary.append("The text demonstrates standard journalistic integrity with neutral emotional distinctness.")
            if analysis['emotional_intensity']['level'] == 'Low':
                summary.append("Emotional drift is minimal, suggesting an objective reporting style.")
                
        # Closing Advice
        if verdict == "Fake":
            summary.append("RECOMMENDATION: Do not share without secondary verification. Isolate and cross-reference claims.")
        else:
            summary.append("RECOMMENDATION: Content appears safe, but always verify specific quotes.")
            
        return " ".join(summary)
