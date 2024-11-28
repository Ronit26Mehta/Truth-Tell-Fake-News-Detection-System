from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = FastAPI()

# Load BERT model
class BERTModel:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, model_path=r'C:\truthtell_Tech4Stack\bert_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded from:", model_path)

    def predict(self, text, max_length=128):
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]

        return predicted_class

# Load the BERT model
bert_model = BERTModel(model_path=r'C:\truthtell_Tech4Stack\bert_model.pth')  # Path to your saved BERT model

# Preprocessor Class for text cleaning (same as in the original code)
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        tokens = word_tokenize(text)  # Tokenize
        tokens = [token for token in tokens if token not in self.stop_words]  # Remove stopwords
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
        return ' '.join(tokens)

# Request body for user input
class PredictionRequest(BaseModel):
    text: str  # Text input for prediction

# Prediction endpoint for BERT Model
@app.post("/predict_bert")
def predict_bert(user_input: PredictionRequest):
    """
    Predict text classification using BERT model.
    Args:
        user_input (PredictionRequest): JSON object containing 'text'.
    Returns:
        A dictionary with predicted class label.
    """
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(user_input.text)

    # Make prediction
    predicted_label = bert_model.predict(cleaned_text)

    return {"text": user_input.text, "predicted_label": predicted_label}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8003)
