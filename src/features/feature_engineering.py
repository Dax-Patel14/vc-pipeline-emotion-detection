import pandas as pd
import numpy as np
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# Logging configuration
logger = logging.getLogger("feature_engineering_logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("feature_engineering.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> int:
    """Loads max_features parameter from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Loaded max_features: {max_features} from {params_path}")
        return max_features
    except FileNotFoundError:
        logger.error(f"Parameter file '{params_path}' not found. Using default max_features=5000")
        return 5000  # Default value
    except KeyError:
        logger.error(f"Missing 'feature_engineering' or 'max_features' in '{params_path}'. Using default max_features=5000")
        return 5000
    except Exception as e:
        logger.exception(f"Unexpected error while loading parameters: {e}")
        return 5000

def read_data(train_path: str, test_path: str):
    """Reads processed train and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        return train_data, test_data
    except FileNotFoundError:
        logger.error("Train or test file not found. Ensure preprocessing step was completed.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.exception(f"Unexpected error while reading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def apply_bow(train_data, test_data, max_features):
    """Applies Bag of Words (BoW) transformation using CountVectorizer."""
    try:
        if train_data.empty or test_data.empty:
            raise ValueError("One or both datasets are empty. Cannot apply BoW.")
        
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        logger.info("BoW transformation applied successfully.")
        return train_df, test_df
    except KeyError:
        logger.error("Required columns 'content' or 'sentiment' not found in dataset.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.exception(f"Unexpected error during BoW transformation: {e}")
        return pd.DataFrame(), pd.DataFrame()

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Saves transformed train and test data to CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        logger.info(f"Feature-engineered data saved successfully in {data_path}")
    except Exception as e:
        logger.exception(f"Error while saving data: {e}")

def main():
    """Main execution function."""
    max_features = load_params('params.yaml')
    train_data, test_data = read_data('./data/interim/train_processed.csv', './data/interim/test_processed.csv')
    
    if train_data.empty or test_data.empty:
        logger.error("Processed train or test data is empty. Exiting.")
        return
    
    train_df, test_df = apply_bow(train_data, test_data, max_features)
    if train_df.empty or test_df.empty:
        logger.error("BoW transformation failed. Exiting.")
        return
    
    save_data(os.path.join("data", "processed"), train_df, test_df)

if __name__ == "__main__":
    main()

    