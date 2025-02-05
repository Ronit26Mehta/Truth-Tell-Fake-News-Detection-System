import streamlit as st
import requests
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

BASE_URL = "http://localhost:8002"  

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

def homepage():
    st.markdown(
        """
        <style>
        .welcome-text {
            font-size: 30px;
            font-weight: bold;
            color: 
            text-align: center;
        }
        .app-description {
            font-size: 18px;
            text-align: center;
            color: 
        }
        .container {
            margin-top: 50px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h2 class="welcome-text">Welcome to the Fake News Classifier App!</h2>', unsafe_allow_html=True)
    st.markdown('<p class="app-description">This app allows you to classify news articles as Fake or Real using machine learning models.</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-description">Choose a model, enter your text, and see the prediction results along with insightful visualizations.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def model_selection_page():
    st.markdown(
        """
        <style>
        .model-selection-container {
            padding: 20px;
            border-radius: 10px;
            background-color: 
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="model-selection-container">', unsafe_allow_html=True)
    st.title("Select the Model")
    model = st.selectbox(
        "Choose the model for classification",
        ["Logistic Regression", "Naive Bayes", "SVM"]
    )
    if model:
        st.session_state.selected_model = model
        st.write(f"You have selected the **{model}** model.")
    st.markdown('</div>', unsafe_allow_html=True)

def model_input_page():
    st.title(f"{st.session_state.selected_model} Model")
    st.write("Enter the text you want to classify:")
    user_input = st.text_area("Text input", height=200)

    if user_input:
        
        if st.session_state.selected_model == "Logistic Regression":
            response = requests.post(f"{BASE_URL}/predict_logistic", json={"text": user_input})
        elif st.session_state.selected_model == "Naive Bayes":
            response = requests.post(f"{BASE_URL}/predict_naive_bayes", json={"text": user_input})
        elif st.session_state.selected_model == "SVM":
            response = requests.post(f"{BASE_URL}/predict_svm", json={"text": user_input})

        if response.status_code == 200:
            prediction = response.json()["predicted_label"]
            st.write(f"Predicted label: {prediction}")
            
            st.subheader("Word Cloud")
            wordcloud = WordCloud(width=600, height=400, background_color='white').generate(user_input)
            fig_wordcloud = plt.figure(figsize=(8, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig_wordcloud)

            data = {"Model": [st.session_state.selected_model], "Prediction": [prediction]}
            df = pd.DataFrame(data)
            fig = px.bar(df, x="Model", y="Prediction", title="Model Prediction")
            st.plotly_chart(fig)

            tfidf_vectorizer = TfidfVectorizer(max_features=10)
            tfidf_matrix = tfidf_vectorizer.fit_transform([user_input])
            tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            features = tfidf_vectorizer.get_feature_names_out()
            tfidf_data = pd.DataFrame(list(zip(features, tfidf_scores)), columns=["Feature", "TF-IDF Score"])
            fig_tfidf = px.bar(tfidf_data, x="Feature", y="TF-IDF Score", title="Top 10 TF-IDF Features")
            st.plotly_chart(fig_tfidf)

            text_length = len(user_input.split())
            st.subheader("Text Length Distribution")
            st.write(f"Number of words in input text: {text_length}")
            fig_length = px.histogram([text_length], title="Text Length Distribution", labels={'value': 'Text Length (Words)'})
            st.plotly_chart(fig_length)

        else:
            st.error("Error in prediction, please try again!")

def main():
    st.set_page_config(page_title="Fake News Classifier App", layout="wide")
    page = st.sidebar.radio("Select a Page", ("Home", "Model Selection", "Model Input"))

    if page == "Home":
        homepage()
    elif page == "Model Selection":
        model_selection_page()
    elif page == "Model Input":
        if "selected_model" in st.session_state:
            model_input_page()
        else:
            st.error("Please select a model first!")

if __name__ == "__main__":
    main()