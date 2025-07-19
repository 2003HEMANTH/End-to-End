import os
import sys
import pandas as pd
import warnings

from src.exception import CustomException
from src.utils import load_object

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

class PredictPipeline:
    def __init__(self):
        pass

    def get_default_values(self):
        return {
            "bedrooms": 3,
            "bathrooms": 2,
            "livingArea": 1200,
            "price": 250000,
            "rentZestimate": 1500,
            "pageViewCount": 0,
            "favoriteCount": 0,
            "propertyTaxRate": 1.2,
            "timeOnZillow": 48,
            "yearBuilt": 2005,
            "homeStatus": "FOR_SALE",
            "homeType": "Single Family",
            "city": "San Jose",
            "zipcode": "95123",
            "state": "CA"
        }

    def add_missing_columns(self, df: pd.DataFrame, expected_columns: list):
        defaults = self.get_default_values()
        for col in expected_columns:
            if col not in df.columns:
                df[col] = defaults.get(col, 0)
        return df

    def predict(self, features: pd.DataFrame):
        try:
            # ✅ Use correct model file name (update if needed)
            model_path = os.path.join("artifacts", "gradient_boosting_roi_model.pkl")  # ✅ correct
 # <-- Rename file if needed
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("✅ Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("✅ Successfully loaded model and preprocessor")

            # ✅ Ensure all required columns are present
            expected_columns = list(preprocessor.feature_names_in_)
            features = self.add_missing_columns(features, expected_columns)
            features = features.reindex(columns=expected_columns, fill_value=0)

            print("✅ Transforming features...")
            data_scaled = preprocessor.transform(features)

            print("✅ Predicting...")
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        bedrooms: float,
        bathrooms: float,
        livingArea: float,
        price: float,
        rentZestimate: float,
        pageViewCount: float = 0,
        favoriteCount: float = 0,
        propertyTaxRate: float = 1.0,
        timeOnZillow: float = 48,
        yearBuilt: float = 2000,
        homeStatus: str = "FOR_SALE",
        homeType: str = "Single Family",
        city: str = "San Jose",
        zipcode: str = "95123",
        state: str = "CA"
    ):
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.livingArea = livingArea
        self.price = price
        self.rentZestimate = rentZestimate
        self.pageViewCount = pageViewCount
        self.favoriteCount = favoriteCount
        self.propertyTaxRate = propertyTaxRate
        self.timeOnZillow = timeOnZillow
        self.yearBuilt = yearBuilt
        self.homeStatus = homeStatus
        self.homeType = homeType
        self.city = city
        self.zipcode = zipcode
        self.state = state

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "bedrooms": [self.bedrooms],
                "bathrooms": [self.bathrooms],
                "livingArea": [self.livingArea],
                "price": [self.price],
                "rentZestimate": [self.rentZestimate],
                "pageViewCount": [self.pageViewCount],
                "favoriteCount": [self.favoriteCount],
                "propertyTaxRate": [self.propertyTaxRate],
                "timeOnZillow": [self.timeOnZillow],
                "yearBuilt": [self.yearBuilt],
                "homeStatus": [self.homeStatus],
                "homeType": [self.homeType],
                "city": [self.city],
                "zipcode": [self.zipcode],
                "state": [self.state],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
