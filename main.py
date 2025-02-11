import os
from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues for Matplotlib
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Fix Permission Issues: Set Writable Directories for Hugging Face & Matplotlib
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["MPLCONFIGDIR"] = "/tmp"

# Create directories if they don’t exist
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

app = Flask(__name__)

# Load Model from Hugging Face (Ensure the model is accessible)
MODEL_NAME = "philipobiorah/bert-imdb-model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

# Function to Predict Sentiment
def predict_sentiment(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]

    sentiments = []
    for chunk in chunks:
        inputs = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        sentiments.append(outputs.logits.argmax(dim=1).item())

    majority_sentiment = Counter(sentiments).most_common(1)[0][0]
    return 'Positive' if majority_sentiment == 1 else 'Negative'

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('upload.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
