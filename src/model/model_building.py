import os
import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from sklearn.ensemble import GradientBoostingClassifier

# Logging configuration
logger = logging.getLogger('model_building_logger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Loads model parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)['model_building']
        logger.info("Successfully loaded model parameters.")
        return params
    except FileNotFoundError:
        logger.error(f"Parameters file '{params_path}' not found.")
        raise
    except KeyError:
        logger.error("Missing 'model_building' section in parameters file.")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while loading parameters: {e}")
        raise

def load_data(train_path: str) -> tuple:
    """Loads training data from CSV file."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info(f"Training data loaded successfully from {train_path}, Shape: {train_data.shape}")
        return X_train, y_train
    except FileNotFoundError:
        logger.error(f"Training data file '{train_path}' not found.")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while loading training data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Trains the Gradient Boosting Classifier."""
    try:
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return clf
    except Exception as e:
        logger.exception(f"Error occurred during model training: {e}")
        raise

def save_model(model: GradientBoostingClassifier, model_dir: str) -> None:
    """Saves the trained model inside 'models/' directory."""
    try:
        os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logger.exception(f"Error occurred while saving the model: {e}")
        raise

def main():
    """Main execution function."""
    params = load_params('params.yaml')
    X_train, y_train = load_data('./data/processed/train_tfidf.csv')
    model = train_model(X_train, y_train, params)
    save_model(model, 'models')  # Save inside 'models/' directory

if __name__ == "__main__":
    main()
