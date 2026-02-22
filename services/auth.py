from fastapi import Request, HTTPException, Depends
from services.api_key_manager import APIKeyManager
from services.billing import StripeBillingService
from database import db
import time

# Initialize Services
api_manager = APIKeyManager(db)
billing_service = StripeBillingService()

async def verify_api_key(request: Request) -> dict:
    """
    Validates API key from X-API-Key header. 
    Returns the organization data if valid.
    """
    api_key = request.headers.get("X-API-Key")
    
    # Bypass for dev console
    if request.url.path.startswith("/api/docs") or request.url.path.startswith("/api/redoc"):
        return {"org_id": "dev", "tier": "enterprise"}
        
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
        
    org_data = api_manager.validate_key(api_key)
    if not org_data:
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    return org_data

async def rate_limit_check(request: Request, org_data: dict = Depends(verify_api_key)):
    """
    Middleware that checks Redis for the organization's current usage.
    Enforces the SaaS tier RPM (Requests Per Minute) limits.
    """
    org_id = org_data["org_id"]
    tier = org_data["tier"]
    limits = billing_service.get_tier_limits(tier)
    
    redis_client = db.get_redis()
    
    if not redis_client:
        # Fail open if Redis is down, log alert. 
        # Don't punish paying customers for internal cache failure.
        return org_data
        
    current_minute = int(time.time() / 60)
    redis_key = f"rate_limit:{org_id}:{current_minute}"
    
    # Atomic increment and expire
    p = redis_client.pipeline()
    p.incr(redis_key)
    p.expire(redis_key, 60)
    result = p.execute()
    
    current_requests = result[0]
    
    if current_requests > limits["rpm_limit"]:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Your current {tier.upper()} tier allows {limits['rpm_limit']} requests per minute. Upgrade for higher limits.",
            headers={"Retry-After": "60"}
        )
        
    return org_data

def require_feature(feature_name: str):
    """
    Factory dependency to restrict specialized endpoints (like RAG or Finetuning)
    to premium tiers.
    """
    async def _require_feature(org_data: dict = Depends(verify_api_key)):
        tier = org_data["tier"]
        has_access = billing_service.check_feature_access(tier, feature_name)
        
        if not has_access:
            raise HTTPException(
                status_code=403, 
                detail=f"Feature '{feature_name}' requires an upgraded subscription tier. Current tier: {tier.upper()}"
            )
        return org_data
    return _require_feature
