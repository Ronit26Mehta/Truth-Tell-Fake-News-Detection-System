from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import torch
from config import Config

class ModelRegistry:
    def __init__(self):
        self.models = {
            "logistic_regression": joblib.load("models/logistic_model.pkl"),
            "naive_bayes": joblib.load("models/naive_bayes_model.pkl"),
            "svm": joblib.load("models/svm_model.pkl"),
            "bert": self.load_bert_model()
        }

    def load_bert_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        ).to(device)
        model.load_state_dict(torch.load(Config.BERT_MODEL_PATH))
        model.eval()
        return model

    def get_model(self, model_name):
        return self.models.get(model_name)