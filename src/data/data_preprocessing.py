import pandas as pd
import numpy as np
import os
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logger = logging.getLogger("data_preprocessing_logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Text Processing Functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    text = re.sub(r'[\W_]+', ' ', text)  # Remove punctuation and special characters
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocess_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

def normalize_text(df, column_name='content'):
    try:
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")
        df[column_name] = df[column_name].astype(str).apply(preprocess_text)
        return df
    except Exception as e:
        logger.exception(f"Error while normalizing text: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}, Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.exception(f"Error while loading data: {e}")
        return pd.DataFrame()

def save_data(df, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data successfully saved at {file_path}")
    except Exception as e:
        logger.exception(f"Error while saving data: {e}")

def main():
    train_data = load_data("./data/raw/train.csv")
    test_data = load_data("./data/raw/test.csv")

    if train_data.empty or test_data.empty:
        logger.error("One or both datasets are empty. Exiting program.")
        return

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    if train_processed_data.empty or test_processed_data.empty:
        logger.error("Text normalization resulted in empty dataset. Exiting program.")
        return

    save_data(train_processed_data, "./data/interim/train_processed.csv")
    save_data(test_processed_data, "./data/interim/test_processed.csv")

if __name__ == "__main__":
    main()
