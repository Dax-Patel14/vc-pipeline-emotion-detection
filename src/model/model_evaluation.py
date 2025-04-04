import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_evaluation")

def load_model(model_path: str):
    """Loads the trained model from a file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Model file '{model_path}' not found.")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error while loading the model: {e}")
        return None

def load_test_data(test_data_path: str):
    """Loads the test dataset from a CSV file."""
    try:
        df = pd.read_csv(test_data_path)
        logger.info(f"Test data loaded successfully from {test_data_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Test data file '{test_data_path}' not found.")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error while loading test data: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate Performance Metrics
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.info("Model evaluation completed successfully.")
        return metrics_dict
    except Exception as e:
        logger.exception(f"Error during model evaluation: {e}")
        return {}

def save_metrics(metrics: dict, output_path: str):
    """Saves the evaluation metrics to a JSON file."""
    try:
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Evaluation metrics saved to {output_path}")
    except Exception as e:
        logger.exception(f"Error while saving metrics: {e}")

def main():
    model_path = 'models\model.pkl'
    test_data_path = './data/processed/test_tfidf.csv'
    metrics_output_path = './reports/metrics.json'

    model = load_model(model_path)
    if model is None:
        return

    test_data = load_test_data(test_data_path)
    if test_data is None:
        return

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_output_path)

if __name__ == "__main__":
    main()
