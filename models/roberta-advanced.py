import pandas as pd
import numpy as np
import torch
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from py2neo import Graph
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time

# Neo4j connection
neo4j_graph = Graph("bolt://localhost:7687", auth=("neo4j", "fakenews"))

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NewsClassifier:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)

    def prepare_data(self, texts, labels=None, max_length=512):
        print(f"Preparing data for {len(texts)} examples...")
        if labels is not None:
            valid_pairs = [(text, label) for text, label in zip(texts, labels) 
                          if text and isinstance(text, str) and text.strip()]
            texts, labels = zip(*valid_pairs)
        
        print("Tokenizing texts...")
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        print("Tokenization complete")

        if labels is not None:
            return NewsDataset(encodings, labels)
        return encodings

    def train(self, train_dataset, eval_dataset=None, epochs=3, batch_size=8):
        print(f"\nStarting training with {len(train_dataset)} examples")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_batches = len(train_dataset) // batch_size + (1 if len(train_dataset) % batch_size != 0 else 0)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.model.train()
            total_loss = 0
            batch_count = 0
            epoch_start = time.time()

            for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processing batch {batch_count}/{total_batches}")
                
                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_count % 50 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"Current average loss: {avg_loss:.4f}")

            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / total_batches
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - Average Loss: {avg_loss:.4f}")

            if eval_dataset:
                print("\nRunning evaluation...")
                accuracy = self.evaluate(eval_dataset)
                print(f"Evaluation Accuracy: {accuracy:.4f}")

        print("\nSaving model...")
        with open('roberta_news_classifier.pkl', 'wb') as f:
            pickle.dump(self.model.state_dict(), f)
        print("Model saved successfully")

    def evaluate(self, eval_dataset):
        self.model.eval()
        total_acc = 0
        total_count = 0
        batch_size = 160
        total_batches = len(eval_dataset) // batch_size + (1 if len(eval_dataset) % batch_size != 0 else 0)
        batch_count = 0

        with torch.no_grad():
            for batch in DataLoader(eval_dataset, batch_size=batch_size):
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Evaluating batch {batch_count}/{total_batches}")
                
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_acc += (predictions == inputs['labels']).sum().item()
                total_count += len(predictions)

        accuracy = total_acc / total_count
        return accuracy
    
if __name__ == "__main__":
    def fetch_news_from_csv(file_path):
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows from CSV")
        return df

    def fetch_news_from_neo4j():
        query = """
        MATCH (n:News)
        RETURN n.title AS title, n.summary AS summary, n.label AS label
        """
        result = neo4j_graph.run(query)
        data = [dict(record) for record in result]
        df = pd.DataFrame(data)
        print(f"Successfully loaded {len(df)} rows from Neo4j")
        return df

    # Load and combine data
    csv_data = fetch_news_from_csv("news_data.csv")
    neo4j_data = fetch_news_from_neo4j()

    # Print sample of labels
    print("\nCSV Labels Sample:")
    print(csv_data['label'].value_counts().head())

    # Create combined dataset
    news_data = pd.concat([csv_data, neo4j_data], ignore_index=True)
    news_data = news_data.dropna(subset=['title', 'summary', 'label'])

    # Create text input
    texts = news_data.apply(
        lambda x: f"{str(x['title']).strip()} {str(x['summary']).strip()}", 
        axis=1
    ).tolist()

    # Label mapping based on the provided categories
    label_mapping = {
        "truth": 4,
        "mostly truth": 3,
        "neutral": 2,
        "mostly fake": 1,
        "fake news": 0
    }

    # Convert labels
    labels = news_data['label'].map(label_mapping)
    
    print("\nLabel distribution:")
    print(labels.value_counts(dropna=False))

    valid_indices = labels.notna()
    valid_texts = [text for i, text in enumerate(texts) if valid_indices[i]]
    valid_labels = labels[valid_indices].astype(int).tolist()

    print(f"\nValid text-label pairs: {len(valid_texts)}")

    if len(valid_texts) == 0:
        raise ValueError("No valid labeled data found after preprocessing")

    # Train classifier
    classifier = NewsClassifier()
    dataset = classifier.prepare_data(valid_texts, valid_labels)

    train_size = int(0.8 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    classifier.train(train_data, test_data, epochs=3)
       