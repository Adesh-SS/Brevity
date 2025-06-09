# Brevity Summarizer

This repository contains the development code for the summarization model used in the Brevity application. It leverages **Hugging Face Transformers** and **PyTorch** to perform abstractive summarization using a fine-tuned version of **DistilBART**.

---

## 🧠 Model Description

The model is a fine-tuned version of `sshleifer/distilbart-cnn-12-6`, trained on a subset of the CNN/DailyMail dataset for abstractive summarization. It receives a long text input (like an article) and generates a concise summary.

### 🔧 Key Features

- Uses `transformers` library (v4.52.4)
- Supports GPU and CPU execution
- Integrated with a Flask backend for inference
- Accepts input in paragraph/text format via POST request
- Returns high-quality summaries in real-time

---

## 🛠 Setup Instructions

Follow these steps to get the project running locally:

### 1. Clone the repository
### 2. Create Virtual Environment ( Windows):
```bash
python -m venv <environment name goes here>
<environment name goes here>\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the cells in the brevity jupeter notebook ( The model code here is coded in GPU Computer)
### 5. Save the model in the name you want
### 6. Run the summarizer file (This runs both in CPU and GPU computer)
```bash
python summarizer.py
```

The summarizer file sends the summary of the text as response to the request provided.

Note: The Model summary token length is only 1024
