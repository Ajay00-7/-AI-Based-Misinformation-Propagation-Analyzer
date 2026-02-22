// Background scripts for context menus and API routing
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "verifyClaim",
        title: "Verify this claim with AI",
        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "verifyClaim") {
        const selectedText = info.selectionText;
        
        // Let the content script know we want to show a loading overlay
        chrome.tabs.sendMessage(tab.id, { 
            action: "showLoading", 
            text: selectedText 
        });

        // Call our local FastAPI/Flask endpoint
        fetch('http://127.0.0.1:5000/api/extension/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: selectedText })
        })
        .then(response => response.json())
        .then(data => {
            // Send the analysis result back to the tab to display
            chrome.tabs.sendMessage(tab.id, { 
                action: "showResult", 
                result: data 
            });
        })
        .catch(error => {
            console.error('Error fetching analysis:', error);
            chrome.tabs.sendMessage(tab.id, { 
                action: "showError", 
                error: error.message 
            });
        });
    }
});
