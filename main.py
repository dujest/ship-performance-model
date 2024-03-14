from predictor import PerformancePredictor
from plotter import plot_fuel_speed_curve
import pandas as pd

if __name__ == "__main__":
    # Create a predictor instance and load the model
    predictor = PerformancePredictor(model_path='spm.bin')  # Load model from file

    # Predict consumption from a CSV file of input parameters
    predictions = predictor.predict_from_csv('input_parameters.csv')

    # Plot the fuel consumption vs. speed curve
    input_parameters = pd.read_csv('input_parameters.csv')
    plot_fuel_speed_curve(input_parameters, predictions)
