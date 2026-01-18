import requests
import json

try:
    url = 'http://127.0.0.1:5000/analyze/text'
    payload = {'text': 'This is a test article to verify the API connection.'}
    headers = {'Content-Type': 'application/json'}
    
    print(f"Sending POST request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"Error: {e}")
