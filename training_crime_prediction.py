import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from prediction import CrimeAnalysis  # Import the CrimeAnalysis class

def train_crime_prediction_model(crime_data, model_path="crime_model.pkl", scaler_path="scaler.pkl", encoder_path="encoder.pkl"):
    """Train a crime prediction model and save it."""
    try:
        # Feature selection and data preparation
        features = ['Latitude', 'Longitude', 'Population Density (per sq. km)', 'Average Income (INR)', 'Unemployment Rate (%)']
        X = crime_data[features].fillna(crime_data[features].mean())  # Handle missing values
        y = crime_data['Crime_Type']  

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Encode categorical target variable
        encoder = OrdinalEncoder()
        y_train = encoder.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = encoder.transform(y_test.values.reshape(-1, 1)).flatten()

        # Train a RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy}")

        # Save the model, scaler, and encoder
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoder, encoder_path)

        print("Crime prediction model trained and saved.")

    except Exception as e:
        print(f"Error during crime prediction model training: {e}")
        raise

def main():
    # Create a CrimeAnalysis object
    analyzer = CrimeAnalysis()

    # Load and preprocess the data
    analyzer.load_and_preprocess_data()

    # Train the crime prediction model
    train_crime_prediction_model(analyzer.crime_data)  # Pass the crime data to the training function

    print("Starting training script...")
    analyzer = CrimeAnalysis()
    print("Loading and preprocessing data...")
    analyzer.load_and_preprocess_data()
    print("Data loaded successfully.")

    print("Training model now...")
    train_crime_prediction_model(analyzer.crime_data)
    print("Model training complete.")

if __name__ == "__main__":
    main()

    print("Running training_crime_prediction.py")
    main()