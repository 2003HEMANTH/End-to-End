import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "gradient_boosting_roi_model.pkl")
    model_features_path: str = os.path.join("artifacts", "model_features.pkl")
    best_model_name_path: str = os.path.join("artifacts", "best_model_name.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("🔄 Splitting train and test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "GradientBoosting": GradientBoostingRegressor()
            }

            report = {}
            best_model = None
            best_score = -np.inf
            best_model_name = None

            for name, model in models.items():
                logging.info(f"⚡ Training model: {name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores = evaluate_model(y_test, y_pred)
                report[name] = scores["R2_Score"]

                if scores["R2_Score"] > best_score:
                    best_model = model
                    best_score = scores["R2_Score"]
                    best_model_name = name

            logging.info(f"✅ Best model: {best_model_name} with R2 score: {best_score:.4f}")

            # Save best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            # Save features
            save_object(self.model_trainer_config.model_features_path, list(pd.DataFrame(X_train).columns))

            # Save model name
            with open(self.model_trainer_config.best_model_name_path, "w") as f:
                f.write(best_model_name)

            return best_model, best_score

        except Exception as e:
            raise CustomException(e, sys)
