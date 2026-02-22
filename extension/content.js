// Content script injected into every page to handle DOM overlays
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "showLoading") {
        showOverlay(`Analyzing claim: "${request.text.substring(0, 50)}..."\n\nPlease wait while models load...`, 'loading');
    } else if (request.action === "showResult") {
        displayAnalysisResult(request.result);
    } else if (request.action === "showError") {
        showOverlay(`Connection Error: Ensure your local Python backend is running on port 5000.\n\nDetails: ${request.error}`, 'error');
    }
});

function displayAnalysisResult(data) {
    if (data.error) {
        showOverlay(`Analysis Error: ${data.error}`, 'error');
        return;
    }

    const analysis = data.analysis;
    const isFake = analysis.verdict === 'Fake';
    const color = isFake ? '#e74c3c' : '#2ecc71';
    const icon = isFake ? '⚠️ LIKELY MISINFORMATION' : '✅ LIKELY AUTHENTIC';

    let html = `
        <div style="font-family: Arial, sans-serif; text-align: left;">
            <div style="font-weight: bold; color: ${color}; font-size: 16px; border-bottom: 1px solid #ccc; padding-bottom: 10px; margin-bottom: 10px;">
                ${icon}
            </div>
            
            <div style="margin-bottom: 10px;">
                <strong>Confidence:</strong> ${(analysis.confidence * 100).toFixed(1)}% <br>
                <strong>Threat Level:</strong> ${analysis.threat_score} / 100
            </div>
            
            <div style="font-size: 13px; color: #555; background: #f9f9f9; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
                <em>AI Analysis:</em> ${analysis.summary}
            </div>
    `;

    if (analysis.extracted_claims && analysis.extracted_claims.total_claims_found > 0) {
        html += `<strong style="font-size:12px;">Extracted Claims & Evidence:</strong><ul style="font-size: 12px; padding-left: 20px; color:#444;">`;
        analysis.extracted_claims.rag_verification.forEach(rag => {
            const verified = rag.verification.match_found;
            if (verified) {
                html += `<li><strong>Fact Found (${rag.verification.source}):</strong> ${rag.verification.evidence}</li>`;
            } else {
                html += `<li><span style="color:#e74c3c;">No trusted source verified this claim:</span> "${rag.claim}"</li>`;
            }
        });
        html += `</ul>`;
    }

    html += `
            <button onclick="document.getElementById('ai-truth-overlay').remove()" style="margin-top: 15px; width: 100%; padding: 8px; background: #333; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Close Analysis
            </button>
        </div>
    `;

    showOverlay(html, 'success', true);
}

function showOverlay(content, type, isHtml = false) {
    // Remove existing overlay if present
    const existing = document.getElementById('ai-truth-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'ai-truth-overlay';

    Object.assign(overlay.style, {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '320px',
        background: 'white',
        boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
        borderRadius: '8px',
        padding: '20px',
        zIndex: '999999',
        fontSize: '14px',
        color: '#333',
        lineHeight: '1.4'
    });

    if (isHtml) {
        overlay.innerHTML = content;
    } else {
        const textNode = document.createElement('div');
        textNode.style.whiteSpace = 'pre-wrap';
        textNode.innerText = content;
        overlay.appendChild(textNode);

        if (type !== 'loading') {
            const btn = document.createElement('button');
            btn.innerText = 'Close';
            btn.onclick = () => overlay.remove();
            Object.assign(btn.style, {
                marginTop: '10px', width: '100%', padding: '5px'
            });
            overlay.appendChild(btn);
        }
    }

    document.body.appendChild(overlay);
}
