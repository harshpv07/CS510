import requests
import base64
import json
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def test_upload_request():
    # API endpoint
    url = "http://127.0.0.1:5000/upload"
    
    # Sample image path - replace with your actual image path
    image_path = "examples/image1.jpg"
    
    # Sample prompt
    prompt = "I am not very happy. I did not get promoted at work. I feel like I am not good enough. Can you help me with some advice?"
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare request payload
    payload = {
        "image": base64_image,
        "prompt": prompt
    }
    
    # Send POST request
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Print response
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_upload_request()
