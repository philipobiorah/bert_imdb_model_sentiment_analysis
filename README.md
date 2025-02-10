# **BERT Sentiment Analysis - Flask App**
This project is a **Flask-based web application** that performs **sentiment analysis** using a **pre-trained BERT model** hosted on **Hugging Face**. The app can analyze text sentiment from **manual input** or **CSV file uploads**, displaying results in a user-friendly web interface.

## **🚀 Features**
- **Pre-trained BERT Model**: Uses `philipobiorah/bert-imdb-model` from **Hugging Face**.
- **Text & File Input**: Accepts **manual text input** and **CSV uploads** for batch processing.
- **Sentiment Analysis**: Classifies text as **Positive** or **Negative**.
- **Visualization**: Displays sentiment distribution via **bar charts**.
- **Web Interface**: Easy-to-use Flask-based UI.

---

## **🌍 Model Information**
- **Model Name**: [`philipobiorah/bert-imdb-model`](https://huggingface.co/philipobiorah/bert-imdb-model)
- **Hosted on**: [Hugging Face Model Hub](https://huggingface.co/philipobiorah/bert-imdb-model)
- **Base Model**: `bert-base-uncased`
- **Framework**: `transformers` (Hugging Face) + `torch` (PyTorch)

---

## **🛠 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/bert-sentiment-analysis.git
cd bert-sentiment-analysis
