# filepath: backend/validation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Example usage
y_true = [0, 1, 0, 1]
y_pred = [0, 1, 1, 1]
metrics = validate_model(y_true, y_pred)
print(metrics)