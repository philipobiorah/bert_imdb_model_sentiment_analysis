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

# Load Model from Hugging Face
MODEL_NAME = "philipobiorah/bert-imdb-model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

# Function to Predict Sentiment
def predict_sentiment(text):
    if not text.strip():
        return "Neutral"  # Avoid processing empty text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    sentiment = outputs.logits.argmax(dim=1).item()
    return "Positive" if sentiment == 1 else "Negative"

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form.get('text', '').strip()
    
    if not text:
        return render_template('upload.html', sentiment="Error: No text provided!")

    sentiment = predict_sentiment(text)
    return render_template('upload.html', sentiment=sentiment)

@app.route('/uploader', methods=['POST'])
def upload_file_post():
    if 'file' not in request.files:
        return "Error: No file uploaded!", 400

    f = request.files['file']
    if f.filename == '':
        return "Error: No file selected!", 400

    try:
        data = pd.read_csv(f)

        # Ensure 'review' column exists
        if 'review' not in data.columns:
            return "Error: CSV file must contain a 'review' column!", 400

        # Predict sentiment for each review
        data['sentiment'] = data['review'].astype(str).apply(predict_sentiment)

        # Generate summary
        sentiment_counts = data['sentiment'].value_counts().to_dict()
        summary = f"Total Reviews: {len(data)}<br>" \
                  f"Positive: {sentiment_counts.get('Positive', 0)}<br>" \
                  f"Negative: {sentiment_counts.get('Negative', 0)}<br>"

        # Generate sentiment plot
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'blue'])
        ax.set_ylabel('Counts')
        ax.set_title('Sentiment Analysis Summary')

        # Save plot as an image
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig)

        return render_template('result.html', tables=[data.to_html(classes='data')], titles=data.columns.values, summary=summary, plot_url=plot_url)

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)