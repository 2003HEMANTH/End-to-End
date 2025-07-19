import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

def main():
    try:
        logging.info("🚀 Starting training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info(f"✅ Data ingestion completed.\nTrain: {train_data_path}\nTest: {test_data_path}")

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"✅ Data transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Model Training
        trainer = ModelTrainer()
        best_model, final_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"✅ Model training completed. Final R² score: {final_accuracy:.4f}")
        print(f"\n🎯 Final R² Score: {final_accuracy:.4f}")

        # ✅ Save model correctly
        model_path = os.path.join("artifacts", "gradient_boosting_roi_model.pkl")
        save_object(
            file_path=model_path,
            obj=best_model
        )
        logging.info(f"✅ Model saved successfully at {model_path}")

    except Exception as e:
        logging.error("❌ Training pipeline failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
