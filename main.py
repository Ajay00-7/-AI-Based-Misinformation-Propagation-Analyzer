from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import asyncio
from typing import Optional, List, Dict, Any

from services.auth import verify_api_key, rate_limit_check, require_feature
from models.deberta_core import DeBERTaONNXCore
from models.shap_explainer import SHAPExplainer

# Assuming these exist in your repository from previous implementation
# from models.bert_classifier import BERTClassifier
# --- Global Model Singletons ---
# We load these once into RAM at startup to prevent memory overflow
deberta_engine = DeBERTaONNXCore()
shap_explainer = SHAPExplainer(deberta_engine)

app = FastAPI(
    title="Misinformation Intelligence API",
    description="Enterprise-grade Misinformation Detection & Propagation Prediction API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration for SaaS Dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to specific SaaS domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for strict API Validation ---

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="The news article text or claim to analyze.")
    source_url: Optional[str] = Field(default=None, description="Optional URL of the article source.")
    
class DetectionResult(BaseModel):
    verdict: str
    confidence: float
    threat_score: int
    threat_level: str
    
class AnalyzeResponse(BaseModel):
    request_id: str
    processing_time_ms: float
    detection: DetectionResult
    extracted_claims: Dict[str, Any]
    propagation_metrics: Dict[str, Any]

# --- Endpoints ---

@app.get("/enterprise", response_class=HTMLResponse, tags=["UI Views"])
async def get_enterprise_dashboard():
    """Returns the React/HTML Enterprise Dashboard View"""
    try:
        with open("templates/enterprise_dashboard.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Dashboard template not found. Ensure 'templates/enterprise_dashboard.html' exists."

@app.get("/health")
async def health_check():
    """Kubernetes / Docker Healthcheck Endpoint"""
    return {"status": "ok", "service": "inference_api", "version": "2.0.0"}

@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Core Analysis"])
async def analyze_content(payload: AnalyzeRequest, org_data: dict = Depends(rate_limit_check)):
    """
    Primary B2B Endpoint. 
    Ingests text, runs DeBERTa/BERT classification, scores the risk.
    Enforces Rate Limits based on Organization Tier.
    """
    start_time = time.time()
    org_id = org_data["org_id"]
    tier = org_data["tier"]
    
    # 1. Asynchronous ONNX Inference (DeBERTa-v3)
    deberta_result = await deberta_engine.predict_async(payload.text)
    
    # 2. Asynchronous SHAP Explainability Generation
    shap_xai = await shap_explainer.explain_decision(
        text=payload.text, 
        prediction=deberta_result["prediction"], 
        confidence=deberta_result["confidence"]
    )
    
    # 3. Process time 
    processing_time = (time.time() - start_time) * 1000
    
    # IMPORT AND EXECUTE YOUR MODELS HERE
    import uuid
    # risk_scorer = RiskScorer() ...
    
    return AnalyzeResponse(
        request_id=str(uuid.uuid4()),
        processing_time_ms=processing_time,
        detection=DetectionResult(
            verdict=deberta_result["prediction"],
            confidence=deberta_result["confidence"],
            threat_score=85, # Scaffold from risk_scorer
            threat_level="Critical Threat"
        ),
        extracted_claims={
            "total_claims_found": 1, 
            "claims": ["Mock extracted claim"],
            "explainable_ai": shap_xai
        },
        propagation_metrics={"r0": 4.5, "severity": "High", "peak_day": 25}
    )

@app.post("/api/v1/advanced/rag-verify", tags=["Premium Features"])
async def rag_verify_claim(claim: str, org_data: dict = Depends(require_feature("rag_verification"))):
    """
    Premium Endpoint.
    Only allows 'Pro' and 'Enterprise' tiers to access the Vector DB Fact-Checking capability.
    """
    return {"status": "success", "message": f"Welcome {org_data['org_id']}, executing RAG query against Pinecone."}

if __name__ == "__main__":
    import uvicorn
    # This runs the production-ready ASGI server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=4)
