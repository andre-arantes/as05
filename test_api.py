import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"

# Test payload formats
payloads = [
    {"inputs": "Test sentence"},
    {"inputs": ["Test sentence"]}
]

for payload in payloads:
    print(f"\nTesting payload: {payload}")
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("Status:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", e)
        print("Response content:", response.text if 'response' in locals() else "No response")