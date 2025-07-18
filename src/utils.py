import os
import sys
import dill
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves any Python object to disk using dill serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a serialized Python object using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(y_true, y_pred):
    """
    Evaluates a regression model and returns a dictionary of metrics.
    """
    try:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {
            "R2_Score": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        }

    except Exception as e:
        raise CustomException(e, sys)


def print_regression_scores(score_dict, model_name="Model"):
    """
    Nicely prints regression evaluation scores.
    """
    print(f"\n📊 Evaluation Report for: {model_name}")
    print("-" * 40)
    print(f"R² Score   : {score_dict['R2_Score']:.4f}")
    print(f"MAE        : {score_dict['MAE']:.4f}")
    print(f"MSE        : {score_dict['MSE']:.4f}")
    print(f"RMSE       : {score_dict['RMSE']:.4f}")
    print("-" * 40)
