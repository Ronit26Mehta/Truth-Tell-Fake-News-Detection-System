import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from py2neo import Graph

# Neo4j connection
neo4j_graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Fetch news data from Neo4j
def fetch_news_from_neo4j():
    query = """
    MATCH (n:News)
    RETURN n.title AS title, n.summary AS summary, n.publisher AS publisher, n.published_at AS published_at, n.link AS link, n.classification AS classification
    """
    return pd.DataFrame(neo4j_graph.run(query).data())

# Prepare data for model training
def prepare_data(news_data):
    # Combine title and summary to form the full text
    news_data['text'] = news_data['title'] + " " + news_data['summary']
    
    # Map classifications to numeric labels
    classification_map = {"Fake": 0, "Low Accuracy": 1, "Medium Accuracy": 2, "High Accuracy": 3, "Accurate": 4}
    news_data['label'] = news_data['classification'].map(classification_map)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(news_data, test_size=0.2, random_state=42)
    
    return train_data, test_data

# Tokenize the text data using RoBERTa tokenizer
def tokenize_data(data, tokenizer, max_length=512):
    encodings = tokenizer(list(data['text']), truncation=True, padding=True, max_length=max_length)
    return encodings

# Compute metrics for model evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True)
    return {'accuracy': acc, 'report': report}

# Fine-tune the RoBERTa model
def train_model(train_data, test_data):
    # Load the RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
    
    # Tokenize the train and test data
    train_encodings = tokenize_data(train_data, tokenizer)
    test_encodings = tokenize_data(test_data, tokenizer)
    
    # Convert to torch datasets
    class NewsDataset(torch.utils.data.Dataset):
        def _init_(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def _getitem_(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def _len_(self):
            return len(self.labels)
    
    # Create datasets for training and evaluation
    train_dataset = NewsDataset(train_encodings, train_data['label'].values)
    test_dataset = NewsDataset(test_encodings, test_data['label'].values)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )
    
    # Trainer for model training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./fine_tuned_roberta")
    
    return trainer

# Plot metrics during training
def plot_metrics(trainer):
    # Get the training loss and evaluation metrics
    logs = trainer.state.log_history
    train_loss = [log['loss'] for log in logs if 'loss' in log]
    eval_accuracy = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]
    
    # Plotting training loss and evaluation accuracy
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eval_accuracy, label='Evaluation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to fetch data, train the model, and evaluate
def main():
    # Step 1: Fetch news data from Neo4j
    news_data = fetch_news_from_neo4j()
    
    if news_data.empty:
        print("No news data found in the database.")
        return

    # Step 2: Prepare data
    train_data, test_data = prepare_data(news_data)

    # Step 3: Train the model
    trainer = train_model(train_data, test_data)

    # Step 4: Plot metrics
    plot_metrics(trainer)

    # Step 5: Evaluate the model
    predictions, labels = trainer.predict(test_data)
    accuracy = accuracy_score(labels, predictions.argmax(axis=1))
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(labels, predictions.argmax(axis=1)))

if _name_ == '_main_':
    main()