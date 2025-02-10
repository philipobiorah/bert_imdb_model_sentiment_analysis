from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

import base64
from io import BytesIO


app = Flask(__name__)


model_name = "philipobiorah/bert-imdb-model"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_name)


model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sentiment(text):
    # Split the text into chunks of 512 tokens
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokenized_text[i:i + 512] for i in range(0, len(tokenized_text), 512)]

    # Predict sentiment for each chunk
    sentiments = []
    for chunk in chunks:
        # inputs = tokenizer.decode(chunk, skip_special_tokens=True)
        inputs = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)  # Explicitly set clean_up_tokenization_spaces
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiments.append(outputs.logits.argmax(dim=1).item())

    # Aggregate the predictions (majority voting)
    sentiment_counts = Counter(sentiments)
    majority_sentiment = sentiment_counts.most_common(1)[0][0]
    return 'Positive' if majority_sentiment == 1 else 'Negative'

@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return render_template('upload.html', sentiment=sentiment)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_post():
    if request.method == 'POST':
        f = request.files['file']
        data = pd.read_csv(f)

        # Predict sentiment for each review
        data['sentiment'] = data['review'].apply(predict_sentiment)

        # Sentiment Analysis Summary
        sentiment_counts = data['sentiment'].value_counts().to_dict()
        summary = f"Total Reviews: {len(data)}<br>" \
                  f"Positive: {sentiment_counts.get('Positive', 0)}<br>" \
                  f"Negative: {sentiment_counts.get('Negative', 0)}<br>"

        # Generate plot
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'blue'])
        ax.set_ylabel('Counts')
        ax.set_title('Sentiment Analysis Summary')
        
        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)

        # Encode the image in base64 and decode it to UTF-8
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Close the plot to free memory
        plt.close(fig)

        return render_template('result.html', tables=[data.to_html(classes='data')], titles=data.columns.values, summary=summary, plot_url=plot_url)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)



