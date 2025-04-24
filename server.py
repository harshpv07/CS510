from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import torch
from vlm import load_image, model, tokenizer, generation_config

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Check if image is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Check if prompt is in the request
    if 'prompt' not in request.form:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Get the image file and prompt
    image_file = request.files['image']
    prompt = request.form['prompt']
    
    # Validate file has a name
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Two options for processing the image:
        
        # Option 1: Process directly from memory
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Option 2: Save to disk first (if needed for debugging)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        # image_file.save(file_path)
        # image = Image.open(file_path)
        
        # Process image using the VLM model
        # Convert PIL Image to tensor with VLM preprocessing
        transform = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        
        # Add image token to prompt
        full_prompt = f'<image>\n{prompt}'
        
        # Get response from model
        response = model.chat(tokenizer, transform, full_prompt, generation_config)
        
        return jsonify({
            'prompt': prompt,
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback_str
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
