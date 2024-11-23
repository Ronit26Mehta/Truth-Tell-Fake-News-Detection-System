import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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

class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming/lemmatization
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)

class TraditionalMLModel:
    def __init__(self, model_type='logistic'):
        self.preprocessor = TextPreprocessor()
        self.model_type = model_type
        
        # Initialize the appropriate model
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=1000)
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', self.model)
        ])
        
    def train(self, X_train, y_train):
        # Preprocess text
        X_train_cleaned = [self.preprocessor.clean_text(text) for text in X_train]
        
        # Train the pipeline
        self.pipeline.fit(X_train_cleaned, y_train)
    
    def predict(self, X):
        X_cleaned = [self.preprocessor.clean_text(text) for text in X]
        return self.pipeline.predict(X_cleaned)
    
    def predict_proba(self, X):
        if self.model_type != 'svm':
            X_cleaned = [self.preprocessor.clean_text(text) for text in X]
            return self.pipeline.predict_proba(X_cleaned)[:, 1]
        else:
            return None

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
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

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
            
            print(f"Epoch {epoch + 1}, Average loss: {total_loss / len(train_loader):.4f}")
    
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

class FakeNewsDetectionSystem:
    def __init__(self):
        self.traditional_models = {
            'logistic': TraditionalMLModel('logistic'),
            'naive_bayes': TraditionalMLModel('naive_bayes'),
            'svm': TraditionalMLModel('svm')
        }
        self.bert_model = None
    
    def train_traditional_models(self, X_train, y_train):
        for model in self.traditional_models.values():
            print(f"Training {model.model_type} model...")
            model.train(X_train, y_train)
    
    def train_bert(self, X_train, y_train):
        print("Training BERT model...")
        self.bert_model = BERTModel()
        self.bert_model.train(X_train, y_train)
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        predictions = model.predict(X_test)
        
        print(f"\nResults for {model_name}:")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        if hasattr(model, 'predict_proba') and model.predict_proba is not None:
            probas = model.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, probas)
            print(f"\nROC-AUC Score: {auc_score:.4f}")
    
    def evaluate_all_models(self, X_test, y_test):
        for model_name, model in self.traditional_models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        if self.bert_model is not None:
            self.evaluate_model(self.bert_model, X_test, y_test, "BERT")

def main():
    # Example usage with sample data
    # Replace this with your actual dataset loading
    data = {
        'text': [
            'Sample real news article text...',
            'Sample fake news article text...',
            # Add more examples
        ],
        'label': [0, 1]  # 0 for real, 1 for fake
    }
    df = pd.DataFrame(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, 
        df['label'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize the system
    system = FakeNewsDetectionSystem()
    
    # Train all models
    system.train_traditional_models(X_train, y_train)
    system.train_bert(X_train, y_train)
    
    # Evaluate all models
    system.evaluate_all_models(X_test, y_test)

if __name__ == "__main__":
    main()