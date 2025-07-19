import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Convert 'timeOnZillow' (e.g. '1 day', '10 hours') to numeric hours
    def convert_to_hours(self, value):
        try:
            value = str(value).lower()
            if 'day' in value:
                return int(value.split()[0]) * 24
            elif 'hour' in value:
                return int(value.split()[0])
            else:
                return 0
        except:
            return 0

    def initiate_data_ingestion(self):
        logging.info("📥 Starting data ingestion process.")
        try:
            # Load the dataset
            df = pd.read_csv("notebook/data/property_listings.csv")
            logging.info(f"📄 Data loaded successfully with shape {df.shape}")

            # Optionally convert 'timeOnZillow' if present
            if "timeOnZillow" in df.columns:
                df["timeOnZillowHours"] = df["timeOnZillow"].apply(self.convert_to_hours)

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("💾 Raw data saved.")

            # Perform train-test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("🔀 Train-test split complete.")

            # Save split data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False) # type: ignore
            test_df.to_csv(self.ingestion_config.test_data_path, index=False) # type: ignore
            logging.info("💾 Train and test data saved.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
