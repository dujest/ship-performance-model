import numpy as np
import matplotlib.pyplot as plt

# Function for plotting the fuel consumption vs. speed curve
def plot_fuel_speed_curve(parameters, consumption_prediction):
    
    # Fit a polynomial
    gps_speed = parameters['gps_speed'].values
    p = np.poly1d(np.polyfit(gps_speed, consumption_prediction, 2))
    
    # Create subplots
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot original data
    ax.scatter(gps_speed, consumption_prediction, color='blue', label='Predicted Consumption')
    
    # Plot polynomial fit
    gps_speed_sorted = np.linspace(min(gps_speed), max(gps_speed), 100)
    ax.plot(gps_speed_sorted, p(gps_speed_sorted), color='red', linewidth=5, label='Polynomial Fit')
    
    ax.set_xlabel('GPS Speed [kts]')
    ax.set_ylabel('Predicted Consumption [mt/day]')
    ax.set_title("Predicted Consumption vs. GPS Speed\n" \
                 "{}={} rad  {}={} m/s  {}={} rad".format( 
                    parameters.columns[1], parameters.iloc[0, 1],
                    parameters.columns[2], parameters.iloc[0, 2],
                    parameters.columns[3], parameters.iloc[0, 3]
                ))
    ax.legend()
    ax.grid(True)

    fig.savefig('FuelSpeedCurve.png')
    
    plt.show()