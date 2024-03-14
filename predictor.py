import pandas as pd
import numpy as np
import xgboost as xgb

class PerformancePredictor:
    def __init__(self, model_path=None):
        """
        Initializes the PerformancePredictor class.

        Args:
            model_path (str, optional): Path to the saved XGBoost model file (.bin).
                If None, the model will be assumed to be already loaded in the 'model' attribute.
                Defaults to None.
        """
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None  # Placeholder for model if not loaded explicitly

    def load_model(self, model_path):
        """
        Loads an XGBoost model from a given file path.

        Args:
            model_path (str): Path to the saved XGBoost model file (.bin).
        """
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict_from_csv(self, csv_file_path):
        """
        Predicts performance from a CSV file.

        Args:
            csv_file_path (str): Path to the CSV file containing performance data.

        Returns:
            numpy.ndarray: Array containing the predicted performance values.
        """

        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model using 'load_model' before making predictions.")

        data = pd.read_csv(csv_file_path)
        data['wind_u'] = np.cos(data['wind_angle']) * data['wind_speed']
        data['wind_v'] = np.sin(data['wind_angle']) * data['wind_speed']
        X = data[['gps_speed', 'course', 'wind_u', 'wind_v', 'wind_speed', 'wind_angle']].values
        X_dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(X_dmatrix)
        return predictions
