import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessor Class
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

# Load datasets
def load_datasets():
    train_df = pd.read_csv('Datasets/train.csv')
    valid_df = pd.read_csv('Datasets/valid.csv')
    test_df = pd.read_csv('Datasets/test.csv')

    # Combine train and validation datasets
    train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Extract features and labels
    X_train, y_train = train_df['statement'], train_df['label']
    X_test, y_test = test_df['statement'], test_df['label']

    return X_train, X_test, y_train, y_test

# Dataset class for BERT
class BERTDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# BERT Model
class BERTModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
    
    def train(self, X_train, y_train, batch_size=16, epochs=3):
        train_dataset = BERTDataset(X_train, y_train, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    
    def predict(self, X, batch_size=16):
        test_dataset = BERTDataset(X, tokenizer=self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        return np.array(predictions)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"BERT model saved as {filename}")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.device)
        print(f"BERT model loaded from {filename}")

def evaluate_model(y_true, y_pred, model_name):
    print(f"\nEvaluation for {model_name}:")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    # Load datasets
    X_train, X_test, y_train, y_test = load_datasets()

    # BERT Training
    print("\nTraining BERT Model...")
    bert_model = BERTModel()
    bert_model.train(X_train.values, y_train.values)
    bert_model.save("bert_model.pth")

    # BERT Evaluation
    print("\nEvaluating BERT Model...")
    y_pred_bert = bert_model.predict(X_test.values)
    evaluate_model(y_test, y_pred_bert, "BERT")

if __name__ == "__main__":
    main()
