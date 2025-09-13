from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----------------------
# AI Model Initialization
# ----------------------

# Using a more general-purpose model
MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"  # You can change this to any model you prefer
print(f"Loading model: {MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

def ello wor(prompt, max_length=512):
    """Generate text using the loaded model."""
    if not model or not tokenizer:
        return "AI model is not loaded. Please check the server logs."

    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        print(f"Error during text generation: {e}")
        return f"An error occurred while generating text: {e}"

""def generate_image(prompt):
    """Generate image using a free API (replace with your preferred image generation API)"""
    try:
        # This is a placeholder - you'll need to replace with a real image generation API
        # For example, you could use Stable Diffusion or another service
        response = requests.post(
            "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
            headers={"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"},
            json={"inputs": prompt},
        )
        
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        else:
            return None
            ""
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

# ----------------------
# Routes
# ----------------------
@app.route('/api/text_generation', methods=['POST'])
def text_generation():
    data = request.json
    task = data.get('task', 'chat')
    prompt = data.get('prompt', '')
    lang = data.get('lang', '')
    
    # Customize prompt based on task
    if task == 'codegen':
        prompt = f"Generate {lang} code for: {prompt}. Provide only the code without explanations."
    elif task == 'bugfix':
        prompt = f"Fix bugs in this {lang} code: {prompt}. Provide only the corrected code."
    elif task == 'codesum':
        prompt = f"Summarize this {lang} code: {prompt}. Provide a concise summary."
    elif task == 'testgen':
        prompt = f"Generate unit tests for this {lang} code: {prompt}. Provide only the test code."
    elif task == 'classify':
        prompt = f"Classify these requirements into SDLC phases: {prompt}. Provide the classification in a structured format."
    elif task == 'chat':
        prompt = f"As a Sustainable Smart City assistant, answer: {prompt}"
    
    response = generate_text(prompt)
    return jsonify({'text': response})

@app.route('/api/image_generation', methods=['POST'])
def image_generation():
    data = request.json
    prompt = data.get('prompt', '')
    
    image_data = generate_image(prompt)
    if image_data:
        return jsonify({'image': image_data})
    else:
        return jsonify({'error': 'Image generation failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)