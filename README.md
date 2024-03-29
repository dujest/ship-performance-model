# Ship Performance Predictor

Welcome to the Ship Performance Predictor! This tool allows you to predict the fuel consumption of a ship based on various input parameters such as GPS speed, course, wind speed, and wind angle.

## How It Works

The Ship Performance Predictor utilizes a machine learning model trained on historical ship performance data.

<p>
<img src="https://drive.google.com/uc?id=19SmjKxYOLd1hztXdezt5DlAHQn_sF0cc" width="70%" >
</p>

Given a CSV file containing input parameters, the predictor makes predictions on fuel consumption using the trained model.

The predictor performs the following steps:

1. Reads input parameters from a CSV file.
2. Calculates additional parameters such as wind components (wind_u and wind_v) based on wind speed and angle.
3. Utilizes the trained machine learning model to predict fuel consumption.
4. Plots the predicted fuel consumption against GPS speed, providing insights into the ship's performance.

<p>
<img src="https://drive.google.com/uc?id=19C6nl2dpuzCg831oAtzU6cQWOJE3twa_" width="70%" >
</p>

## Installation

To use the Ship Performance Predictor, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/ship-performance-predictor.git
    ```

2. Navigate to the project directory:

    ```bash
    cd ship-performance-predictor
    ```

3. Create a virtual environment using venv:

    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

5. Install the required packages listed in requirements.txt:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have a trained machine learning model ready for prediction.

2. Prepare a CSV file containing input parameters. The CSV file should include columns for:

    - `gps_speed` (Unit: [kts], Description: Speed over ground)
    - `course` (Unit: [rad], Description: Vessel course in radians in lon,lat orthographic convention (0 means going E pi/2 N, …))
    - `wind_speed` (Unit: [m/s], Description: Wind speed component)
    - `wind_angle` (Unit: [rad], Description: Wind angle in radians in lon,lat orthographic convention (0 means going E pi/2 N, …))

3. Run the main script:

    ```bash
    python main.py
    ```

    The script will read the input parameters from the CSV file, make predictions using the trained model, and generate a plot showing the predicted fuel consumption vs. GPS speed.

4. Find the generated plot saved as FuelSpeedCurve.png in the project directory.

### Example CSV File Format

Here's an example of how your CSV file should be structured:

```python
gps_speed,course,wind_speed,wind_angle
5,3,15,1
7,3,15,1
10,3,15,1
12,3,15,1
...
