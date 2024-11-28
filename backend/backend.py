from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = FastAPI()

# Load models (adjust paths to match your saved model locations)
logistic_model_path = r"C:\truthtell_Tech4Stack\logistic_model.pkl"
naive_bayes_model_path = r"C:\truthtell_Tech4Stack\naive_bayes_model.pkl"
svm_model_path = r"C:\truthtell_Tech4Stack\svm_model.pkl"

logistic_model = joblib.load(logistic_model_path)
naive_bayes_model = joblib.load(naive_bayes_model_path)
svm_model = joblib.load(svm_model_path)

# Preprocessor Class (same as in the original code)
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        text = text.lower()  # Lowercase
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

# Prediction endpoint for Logistic Regression
@app.post("/predict_logistic")
def predict_logistic(user_input: PredictionRequest):
    """
    Predict text classification using Logistic Regression.
    Args:
        user_input (PredictionRequest): JSON object containing 'text'.
    Returns:
        A dictionary with predicted class label.
    """
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(user_input.text)

    # Convert to DataFrame for model input
    input_data = pd.DataFrame([{'statement': cleaned_text}])

    # Make prediction
    prediction = logistic_model.predict(input_data['statement'])
    prediction_label = int(prediction[0])  # Convert numpy.int64 to Python int

    return {"text": user_input.text, "predicted_label": prediction_label}

# Prediction endpoint for Naive Bayes
@app.post("/predict_naive_bayes")
def predict_naive_bayes(user_input: PredictionRequest):
    """
    Predict text classification using Naive Bayes.
    Args:
        user_input (PredictionRequest): JSON object containing 'text'.
    Returns:
        A dictionary with predicted class label.
    """
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(user_input.text)

    # Convert to DataFrame for model input
    input_data = pd.DataFrame([{'statement': cleaned_text}])

    # Make prediction
    prediction = naive_bayes_model.predict(input_data['statement'])
    prediction_label = int(prediction[0])  # Convert numpy.int64 to Python int

    return {"text": user_input.text, "predicted_label": prediction_label}

# Prediction endpoint for SVM
@app.post("/predict_svm")
def predict_svm(user_input: PredictionRequest):
    """
    Predict text classification using SVM.
    Args:
        user_input (PredictionRequest): JSON object containing 'text'.
    Returns:
        A dictionary with predicted class label.
    """
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(user_input.text)

    # Convert to DataFrame for model input
    input_data = pd.DataFrame([{'statement': cleaned_text}])

    # Make prediction
    prediction = svm_model.predict(input_data['statement'])
    prediction_label = int(prediction[0])  # Convert numpy.int64 to Python int

    return {"text": user_input.text, "predicted_label": prediction_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)
