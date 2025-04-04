import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

import logging

# logging configure
logger = logging.getLogger('data_ingestion_logger')# create a logger object (name is: data_ingestion_logger)
logger.setLevel('DEBUG') #  set basic level

console_handler = logging.StreamHandler() #create console_handler object
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # format of mesage
console_handler.setFormatter(formatter) # Attache formatter with handler
file_handler.setFormatter(formatter)

# Attache handler with logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)




def load_parms(params_path: str) -> float:
    """Loads test_size parameter from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.info(f"Loaded test_size: {test_size} from {params_path}")
        return test_size
    except FileNotFoundError:
        logger.error(f"Parameter file '{params_path}' not found. Using default test_size=0.2")
        return 0.2  # Default test size
    except KeyError:
        logger.error(f"Missing 'data_ingestion' or 'test_size' in '{params_path}'. Using default test_size=0.2")
        return 0.2  # Default test size
    except Exception as e:
        logger.exception(f"Unexpected error while loading parameters: {e}")
        return 0.2  # Default test size

def read_data(url: str) -> pd.DataFrame:
    """Reads CSV data from a URL."""
    try:
        df = pd.read_csv(url)
        logger.info(f"Data successfully read from {url}, shape: {df.shape}")
        return df
    except pd.errors.ParserError:
        logger.error(f"Error: Unable to parse CSV data from '{url}'. Returning empty DataFrame.")
        return pd.DataFrame()  # Return empty DataFrame
    except Exception as e:
        logger.exception(f"Unexpected error while reading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the DataFrame by filtering and modifying sentiment values."""
    try:
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot process data.")

        df.drop(columns=['tweet_id'], inplace=True, errors='ignore')  # Ignore if column is missing
        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])]

        if final_df.empty:
            raise ValueError("Filtered DataFrame is empty. Check sentiment values.")

        final_df['sentiment'].replace({'neutral': 1, 'sadness': 0}, inplace=True)
        logger.info(f"Processed data successfully. Final shape: {final_df.shape}")
        return final_df
    except KeyError:
        logger.error("Error: 'sentiment' column not found in the DataFrame.")
        return pd.DataFrame()  # Return empty DataFrame
    except Exception as e:
        logger.exception(f"Unexpected error while processing data: {e}")
        return pd.DataFrame()  # Return empty DataFrame

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Saves train and test datasets to CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)  # Avoid errors if directory exists

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info(f"Data saved successfully at {data_path}")
    except Exception as e:
        logger.exception(f"Error while saving data: {e}")

def main():
    """Main execution function."""
    test_size = load_parms('params.yaml')
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

    final_df = process_data(df)
    if final_df.empty:
        logger.error("Processed DataFrame is empty. Exiting program.")
        return

    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    logger.info(f"Split data into train ({train_data.shape}) and test ({test_data.shape})")

    data_path = os.path.join("data", "raw") 
    save_data(data_path, train_data, test_data)

if __name__ == "__main__":
    main()
