import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save any Python object to the given file path using dill.
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
    Load any Python object saved using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(true, predicted):
    """
    Returns MAE, RMSE, and R² score.
    """
    try:
        mae = mean_absolute_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2 = r2_score(true, predicted)
        return {"MAE": mae, "RMSE": rmse, "R2_Score": r2}
    except Exception as e:
        raise CustomException(e, sys)
