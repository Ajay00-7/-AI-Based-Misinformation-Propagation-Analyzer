class RiskScorer:
    """
    Calculates a unified 0-100 Threat/Risk Score.
    Combines Model Probability, Emotional Intensity, Claim Density, and Viral Potential.
    """
    
    def calculate_score(self, bert_prob, is_fake, claims_count, emotional_intensity, r0_value):
        """
        Calculates a risk score out of 100.
        
        Args:
            bert_prob (float): Confidence level from BERT model (0.0 - 1.0)
            is_fake (bool): Whether the model classified it as Fake
            claims_count (int): How many verifiable claims were extracted
            emotional_intensity (str): High, Moderate, or Low
            r0_value (float): Reproduction number from SIR model
        """
        if not is_fake:
            # Real news inherently carries low threat risk. 
            # We scale the risk down entirely.
            base_risk = (1.0 - bert_prob) * 20 # Max 20 risk if real.
            return {
                "score": int(base_risk),
                "level": "Low",
                "color_code": "#27ae60" # Green
            }
            
        # If Fake, calculate composite risk
        # 1. Base Model Confidence (40% weight)
        score = bert_prob * 40.0
        
        # 2. Viral Threat (30% weight)
        # R0 typically ranges from 1.0 to 5.0+
        viral_factor = min((r0_value / 6.0) * 30.0, 30.0) 
        score += viral_factor
        
        # 3. Emotional Manipulation (15% weight)
        emotion_map = {"High": 15.0, "Moderate": 8.0, "Low": 2.0}
        score += emotion_map.get(emotional_intensity, 5.0)
        
        # 4. Claim Density Threat (15% weight)
        # More false claims = more damage to public discourse
        claim_factor = min((claims_count / 5.0) * 15.0, 15.0)
        score += claim_factor
        
        score = min(max(int(score), 0), 100)
        
        # Determine strict category
        if score > 75:
            level = "Critical Threat"
            color = "#c0392b" # Red
        elif score > 50:
            level = "Elevated Risk"
            color = "#f39c12" # Orange
        elif score > 25:
            level = "Moderate"
            color = "#f1c40f" # Yellow
        else:
            level = "Low"
            color = "#27ae60" # Green
            
        return {
            "score": score,
            "level": level,
            "color_code": color,
            "components": {
                "model_confidence_weight": int(bert_prob * 40.0),
                "viral_threat_weight": int(viral_factor),
                "emotional_manipulation_weight": int(emotion_map.get(emotional_intensity, 5.0)),
                "claim_damage_weight": int(claim_factor)
            }
        }
