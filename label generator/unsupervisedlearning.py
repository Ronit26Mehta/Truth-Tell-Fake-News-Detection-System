from flask import Flask, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from neo4j import GraphDatabase

# Initialize Flask app
app = Flask(__name__)

# Initialize Neo4j driver
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "fakenews"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Function to fetch news data from CSV
def fetch_news_from_csv(csv_file):
    try:
        return pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

# Function to fetch news data from Neo4j
def fetch_news_from_neo4j():
    query = """
    MATCH (n:News)
    RETURN n.title AS title, n.summary AS summary
    """
    with driver.session() as session:
        result = session.run(query)
        return pd.DataFrame(result.data())

# Function to classify news using KMeans
def classify_news(data):
    # Combine title and summary for feature extraction
    data['text'] = data['title'] + " " + data['summary']
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    features = tfidf.fit_transform(data['text'])
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    
    # Map clusters to labels (0: Fake, ..., 4: Accurate)
    data['classification'] = data['cluster'].map({
        0: "Fake",
        1: "Low Accuracy",
        2: "Medium Accuracy",
        3: "High Accuracy",
        4: "Accurate"
    })
    return data

# Endpoint to classify and store news
@app.route('/classify_news', methods=['GET'])
def classify_and_store_news():
    # Step 1: Fetch news from CSV
    news_data_csv = fetch_news_from_csv("news_data.csv")
    
    # Step 2: Fetch news from Neo4j
    news_data_neo4j = fetch_news_from_neo4j()
    
    # Step 3: Combine both datasets
    news_data = pd.concat([news_data_csv, news_data_neo4j], ignore_index=True)
    
    if news_data.empty:
        return jsonify({"error": "No news data found"}), 404
    
    # Step 4: Classify news articles
    classified_data = classify_news(news_data)
    
    # Step 5: Store results back into CSV
    classified_data.to_csv("classified_news_data.csv", index=False)
    
    return jsonify({"message": "News classified and stored successfully"}), 200

# Endpoint to fetch classified news
@app.route('/get_classified_news', methods=['GET'])
def get_classified_news():
    try:
        df = pd.read_csv("classified_news_data.csv")
        result = df.to_dict(orient="records")
        return jsonify(result)
    except Exception as e:
        print(f"Error reading classified CSV file: {e}")
        return jsonify({"error": f"Error reading classified news data: {e}"}), 500

# Main execution
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)