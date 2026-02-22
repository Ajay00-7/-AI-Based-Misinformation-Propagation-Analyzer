import secrets
import hashlib
from datetime import datetime

class APIKeyManager:
    """
    Manages generation, storage, and validation of B2B SaaS API keys.
    In production, this interfaces directly with the PostgreSQL `users` table.
    """
    def __init__(self, db_manager):
        self.db = db_manager
        
    def generate_key(self, prefix="ms_"):
        """Generates a secure, readable API key format (e.g., ms_1a2b3c...)"""
        random_bytes = secrets.token_hex(24)
        return f"{prefix}{random_bytes}"
        
    def hash_key(self, api_key: str) -> str:
        """Never store plaintext API keys; store SHA-256 hashes."""
        return hashlib.sha256(api_key.encode()).hexdigest()
        
    def create_api_client(self, org_name: str, tier: str = "free"):
        """
        Creates a new organization client, generates an API key, 
        and stores the hashed key in the database.
        """
        raw_key = self.generate_key()
        hashed_key = self.hash_key(raw_key)
        
        # Mock database insertion
        # cursor.execute("INSERT INTO api_clients (org, key_hash, tier) VALUES (%s, %s, %s)",
        #                (org_name, hashed_key, tier))
        
        return {
            "organization": org_name,
            "api_key": raw_key, # Only show this ONCE to the user
            "tier": tier,
            "created_at": datetime.utcnow().isoformat()
        }
        
    def validate_key(self, api_key: str) -> dict:
        """
        Hashes the incoming key and checks against the DB.
        Returns the organization and their billing tier rules if valid.
        """
        # For the mock/MVP, we hardcode some structural rules based on the prefix length
        if not api_key or not api_key.startswith("ms_"):
            return None
            
        hashed_attempt = self.hash_key(api_key)
        
        # Mock database lookup
        # SELECT tier FROM api_clients WHERE key_hash = attempt
        
        # Fake lookup logic for scaffold
        if "enterprise" in api_key:
            return {"org_id": "org_999", "tier": "enterprise"}
        elif "pro" in api_key:
            return {"org_id": "org_888", "tier": "pro"}
        
        return {"org_id": "org_777", "tier": "free"}
