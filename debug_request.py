import requests
import json

try:
    print("Testing API...")
    url = 'http://127.0.0.1:5000/analyze/text'
    data = {'text': 'This is a test article to verify the server is running correctly.'}
    
    response = requests.post(url, json=data, timeout=60)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
except Exception as e:
    print(f"Error: {e}")
