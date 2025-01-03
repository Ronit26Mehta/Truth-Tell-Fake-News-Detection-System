from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import time
from transformers import pipeline
import tensorflow as tf
import os
import logging
from neo4j import GraphDatabase
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from textblob import TextBlob
import csv

# Initialize Flask app
app = Flask(__name__)

# Configure logging to write to a file
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load summarization model
summarizer = pipeline("summarization", framework="tf")

# Initialize Neo4j driver
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "fakenews"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Function to extract symbols from CSVs
def get_symbols_from_csvs(csv_files):
    symbols = set()
    for file in csv_files:
        df = pd.read_csv(file)
        symbols.update(df['SYMBOL'].unique())  # Updated column name to 'SYMBOL'
    return list(symbols)  # Limit to 20 symbols for testing

# Function to summarize news content
def summarize_content(content):
    try:
        if len(content.split()) > 50:  # Summarize only long content
            return summarizer(content, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        return content
    except Exception as e:
        logging.error(f"Error summarizing content: {e}")
        return "Summary not available."

# Function to fetch news using yfinance with retries
def fetch_news_with_yfinance(symbol):
    try:
        stock = yf.Ticker(symbol)
        news = stock.news  # Fetch news articles
        if news is None:
            logging.info(f"No news attribute found for symbol: {symbol}")
            return []
        logging.info(f"Fetched {len(news)} news articles for symbol: {symbol}")
        # Log the fetched news articles
        logging.info(f"Fetched news articles: {news}")
        return news  # Returns a list of news dictionaries
    except Exception as e:
        logging.error(f"Error fetching news for {symbol}: {e}")
        return []

# Function to create nodes and relationships in Neo4j
def create_knowledge_graph(symbol, news_articles):
    with driver.session() as session:
        for article in news_articles:
            content = article.get("content", {})
            title = content.get("title")
            publisher = content.get("provider", {}).get("displayName")
            link = content.get("canonicalUrl", {}).get("url")
            published_at = content.get("pubDate")
            summary = content.get("summary", "")

            if not title:
                logging.warning(f"Skipping article with missing title: {article}")
                continue

            # Analyze sentiment using TextBlob
            sentiment = TextBlob(summary).sentiment
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity

            # Generate label based on polarity
            if polarity > 0.7:
                label = "truth"
            elif 0.2 < polarity <= 0.7:
                label = "mostly truth"
            elif -0.2 < polarity <= 0.2:
                label = "neutral"
            elif -1 < polarity <= -0.2:
                label = "mostly fake"
            else:
                label = "fake news"

            # Store data in CSV
            store_data_in_csv(symbol, title, publisher, link, published_at, summary, polarity, subjectivity, label)

            logging.info(f"Creating stock node for symbol: {symbol}")
            session.write_transaction(create_stock_node, symbol)
            logging.info(f"Creating news node for title: {title}")
            session.write_transaction(create_news_node, title, publisher, link, published_at, summary)
            logging.info(f"Creating relationship between stock: {symbol} and news: {title}")
            session.write_transaction(create_relationship, symbol, title)
            logging.info(f"Successfully created relationship for symbol: {symbol} and news: {title}")

def create_stock_node(tx, symbol):
    try:
        tx.run("MERGE (s:Stock {symbol: $symbol})", symbol=symbol)
        logging.info(f"Stock node created for symbol: {symbol}")
    except Exception as e:
        logging.error(f"Error creating stock node for symbol: {symbol}: {e}")

def create_news_node(tx, title, publisher, link, published_at, summary):
    try:
        tx.run("""
            MERGE (n:News {title: $title})
            SET n.publisher = $publisher, n.link = $link, n.published_at = $published_at, n.summary = $summary
            """, title=title, publisher=publisher, link=link, published_at=published_at, summary=summary)
        logging.info(f"News node created for title: {title}")
    except Exception as e:
        logging.error(f"Error creating news node for title: {title}: {e}")

def create_relationship(tx, symbol, title):
    try:
        tx.run("""
            MATCH (s:Stock {symbol: $symbol}), (n:News {title: $title})
            MERGE (s)-[:HAS_NEWS]->(n)
            """, symbol=symbol, title=title)
        logging.info(f"Relationship created between stock: {symbol} and news: {title}")
    except Exception as e:
        logging.error(f"Error creating relationship between stock: {symbol} and news: {title}: {e}")

# Function to store data in CSV
def store_data_in_csv(symbol, title, publisher, link, published_at, summary, polarity, subjectivity, label):
    file_exists = os.path.isfile('news_data.csv')
    with open('news_data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['symbol', 'title', 'publisher', 'link', 'published_at', 'summary', 'polarity', 'subjectivity', 'label'])
        writer.writerow([symbol, title, publisher, link, published_at, summary, polarity, subjectivity, label])
    logging.info(f"Appended news article with title: {title} to CSV")

# Function to fetch and update news in real-time with retries
def fetch_and_update_news(symbols):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    while True:
        for symbol in symbols:
            logging.info(f"Fetching news for symbol: {symbol}")
            news_articles = fetch_news_with_yfinance(symbol)
            if news_articles:
                logging.info(f"Processing {len(news_articles)} articles for symbol: {symbol}")
                create_knowledge_graph(symbol, news_articles)
            else:
                logging.info(f"No news articles found for symbol: {symbol}")
        time.sleep(60)  # Fetch and update every minute

# Flask routes
@app.route('/get_latest_news', methods=['GET'])
def get_latest_news():
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (n:News)
                RETURN n
                ORDER BY n.published_at DESC
                LIMIT 4
            """)
            news = [record["n"] for record in result]
            return jsonify(news)
    except Exception as e:
        logging.error(f"Error fetching latest news from Neo4j: {e}")
        return jsonify([])

@app.route('/get_all_news', methods=['GET'])
def get_all_news():
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (n:News)
                RETURN n
                ORDER BY n.published_at DESC
            """)
            news = [record["n"] for record in result]
            return jsonify(news)
    except Exception as e:
        logging.error(f"Error fetching all news from Neo4j: {e}")
        return jsonify([])

@app.route('/get_news_by_symbol/<symbol>', methods=['GET'])
def get_news_by_symbol(symbol):
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Stock {symbol: $symbol})-[:HAS_NEWS]->(n:News)
                RETURN n
                ORDER BY n.published_at DESC
            """, symbol=symbol)
            news = [record["n"] for record in result]
            return jsonify(news)
    except Exception as e:
        logging.error(f"Error fetching news by symbol from Neo4j: {e}")
        return jsonify([])

# Main execution
if __name__ == '__main__':
    csv_files = ["Equity.csv", "EQUITY_L.csv", "SME_EQUITY_L.csv"]
    symbols = get_symbols_from_csvs(csv_files)
    
    # Start the real-time news fetching and updating in a separate thread
    import threading
    threading.Thread(target=fetch_and_update_news, args=(symbols,), daemon=True).start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)