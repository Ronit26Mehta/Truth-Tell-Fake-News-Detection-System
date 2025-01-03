# filepath: backend/social_media_analysis.py
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

def identify_echo_chambers(texts):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    return kmeans.labels_

def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Example usage
texts = ["This is a test message", "Another test message"]
clusters = identify_echo_chambers(texts)
sentiments = [sentiment_analysis(text) for text in texts]
print(clusters, sentiments)