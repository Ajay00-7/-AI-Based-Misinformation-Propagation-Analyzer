
import requests
import json
import time

url = 'http://localhost:5000/analyze/text'
data = {'text': 'This is a test article about fake news to verify the backend connection.'}
headers = {'Content-Type': 'application/json'}

print(f"Sending request to {url}...")
try:
    start = time.time()
    response = requests.post(url, json=data, headers=headers, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    print(f"Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Request failed: {e}")
