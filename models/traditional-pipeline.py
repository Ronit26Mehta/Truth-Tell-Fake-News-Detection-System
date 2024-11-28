import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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


def load_datasets():
    train_df = pd.read_csv('Datasets/train.csv')
    valid_df = pd.read_csv('Datasets/valid.csv')
    test_df = pd.read_csv('Datasets/test.csv')

    
    train_df = pd.concat([train_df, valid_df], ignore_index=True)

    
    X_train, y_train = train_df['statement'], train_df['label']
    X_test, y_test = test_df['statement'], test_df['label']

    return X_train, X_test, y_train, y_test


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename):
    return joblib.load(filename)


def evaluate_roc_auc(y_true, probas_or_scores, multi_class=False):
    try:
        if multi_class:
            score = roc_auc_score(y_true, probas_or_scores, multi_class='ovr')
        else:
            score = roc_auc_score(y_true, probas_or_scores)
        print(f"\nROC-AUC Score: {score:.4f}")
    except ValueError as e:
        print(f"\nROC-AUC could not be computed: {e}")


class TraditionalModelPipeline:
    def __init__(self, model_type='logistic'):
        self.preprocessor = TextPreprocessor()
        
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=1000)
        else:
            raise ValueError("Unsupported model type!")

        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', self.model)
        ])

    def train(self, X_train, y_train):
        X_train_cleaned = X_train.apply(self.preprocessor.clean_text)
        self.pipeline.fit(X_train_cleaned, y_train)

    def predict(self, X):
        X_cleaned = X.apply(self.preprocessor.clean_text)
        return self.pipeline.predict(X_cleaned)

    def evaluate(self, X_test, y_test):
        X_test_cleaned = X_test.apply(self.preprocessor.clean_text)
        predictions = self.pipeline.predict(X_test_cleaned)

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
            probas = self.pipeline.predict_proba(X_test_cleaned)[:, 1]
            evaluate_roc_auc(y_test, probas, multi_class=False)
        elif hasattr(self.pipeline.named_steps['classifier'], "decision_function"):
            scores = self.pipeline.decision_function(X_test_cleaned)
            evaluate_roc_auc(y_test, scores, multi_class=len(set(y_test)) > 2)
        else:
            print("\nROC-AUC Score: Not available for this model")


def main():
    
    X_train, X_test, y_train, y_test = load_datasets()

    
    print("\nTraining Logistic Regression Model...")
    logistic_model = TraditionalModelPipeline(model_type='logistic')
    logistic_model.train(X_train, y_train)
    logistic_model.evaluate(X_test, y_test)
    save_model(logistic_model.pipeline, "logistic_model.pkl")

    
    print("\nTraining Naive Bayes Model...")
    nb_model = TraditionalModelPipeline(model_type='naive_bayes')
    nb_model.train(X_train, y_train)
    nb_model.evaluate(X_test, y_test)
    save_model(nb_model.pipeline, "naive_bayes_model.pkl")

    
    print("\nTraining SVM Model...")
    svm_model = TraditionalModelPipeline(model_type='svm')
    svm_model.train(X_train, y_train)
    svm_model.evaluate(X_test, y_test)
    save_model(svm_model.pipeline, "svm_model.pkl")

    
    print("\nLoading Saved Logistic Regression Model...")
    loaded_logistic = load_model("logistic_model.pkl")
    predictions = loaded_logistic.predict(X_test.apply(TextPreprocessor().clean_text))
    print("\nLoaded Model Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
