import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import spmatrix

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df: pd.DataFrame) -> ColumnTransformer:
        try:
            numerical_columns = [
                "bedrooms", "bathrooms", "livingArea", "price", "rentZestimate",
                "pageViewCount", "favoriteCount", "propertyTaxRate",
                "timeOnZillow", "yearBuilt"
            ]
            categorical_columns = []

            for col in ["homeStatus", "homeType", "city", "zipcode", "state"]:
                if col in df.columns:
                    categorical_columns.append(col)

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("✅ Loaded train and test data for transformation")

            drop_cols = ["zpid", "url", "lastUpdated", "streetAddress", "dateSold",
                         "datePosted", "livingAreaUnits", "county"]
            train_df.drop(columns=[col for col in drop_cols if col in train_df.columns], inplace=True, errors='ignore')
            test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], inplace=True, errors='ignore')

            def convert_time(value):
                if pd.isna(value):
                    return np.nan
                value = str(value).strip().lower()
                try:
                    num = int(value.split()[0])
                    if "day" in value:
                        return num * 24
                    elif "hour" in value:
                        return num
                except:
                    return np.nan
                return np.nan

            for df in [train_df, test_df]:
                if "timeOnZillow" in df.columns:
                    df["timeOnZillow"] = df["timeOnZillow"].apply(convert_time)

            for df in [train_df, test_df]:
                df["annual_rent_income"] = df["rentZestimate"] * 12
                df["roi"] = (df["annual_rent_income"] / df["price"]) * 100
                df["roi"] = df["roi"].replace([np.inf, -np.inf], np.nan)
                df.dropna(subset=[
                    "roi", "bedrooms", "bathrooms", "livingArea", "price", "rentZestimate"
                ], inplace=True)

            target_column = "roi"

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object(train_df)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            if isinstance(input_feature_train_arr, spmatrix):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if isinstance(input_feature_test_arr, spmatrix):
                input_feature_test_arr = input_feature_test_arr.toarray()

            logging.info("📦 Feature array shapes:")
            logging.info(f"Train features: {input_feature_train_arr.shape}")
            logging.info(f"Train target: {target_feature_train_df.shape}")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
