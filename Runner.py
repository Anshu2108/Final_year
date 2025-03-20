import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.prediction import CrimeAnalysis
import backend.training_crime_prediction
from frontend.app import launch_gradio_app

def main():
    # Initialize the analyzer with updated file paths
    analyzer = CrimeAnalysis(
        crime_data_path="data set/Crime_Prediction.csv",
        conviction_data_path="data set/Conviction_Rate.csv"
    )

    # Load and preprocess data
    analyzer.load_and_preprocess_data()

    # Train the crime prediction model by calling the training script
    backend.training_crime_prediction.main()

    # Calculate conviction rates
    analyzer.calculate_conviction_rate()

    # Load the trained model
    analyzer.load_model()

    # Launch the Gradio app by calling the frontend/app.py script
    launch_gradio_app(analyzer)

    print("Initializing CrimeAnalysis...")  
    analyzer = CrimeAnalysis(
        crime_data_path="data set/Crime_Prediction.csv",
        conviction_data_path="data set/Conviction_Rate.csv"
    )

    print("Loading and preprocessing data...")  
    analyzer.load_and_preprocess_data()
    print("Data preprocessing complete.")

    print("Training crime prediction model...")
    backend.training_crime_prediction.main()
    print("Model training complete.")

    print("Calculating conviction rates...")
    analyzer.calculate_conviction_rate()
    print("Conviction rate calculation complete.")

    print("Loading trained model...")
    analyzer.load_model()
    print("Model loaded successfully.")

    print("Launching Gradio app...")
    launch_gradio_app(analyzer)
    print("Gradio app launched.")

if __name__ == "__main__":
    main()
