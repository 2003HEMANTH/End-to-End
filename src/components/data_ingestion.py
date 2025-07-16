import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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

    def initiate_data_ingestion(self) -> tuple[str, str]:
        logging.info("🔁 Starting data ingestion process.")
        try:
            # Load dataset
            df: pd.DataFrame = pd.read_csv("notebook/data/property_listings.csv") 
            logging.info("✅ Successfully loaded property_listings.csv.")

            # Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Split dataset
            # train_set: pd.DataFrame
            # test_set: pd.DataFrame
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # type: ignore

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)  # type: ignore
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)    # type: ignore

            logging.info("✅ Data ingestion completed.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("❌ Exception in data ingestion.")
            raise CustomException(e, sys)


# Optional: run standalone
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
