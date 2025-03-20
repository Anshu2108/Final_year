import pandas as pd
import numpy as np
import joblib  # For loading the model
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # Import necessary for loading
from statsmodels.tsa.arima.model import ARIMA 

class CrimeAnalysis:
    def __init__(self, crime_data_path="data set/Crime_Prediction.csv", conviction_data_path="data set/Conviction_Rate.csv",
                 model_path="crime_model.pkl", scaler_path="scaler.pkl", encoder_path="encoder.pkl"):
        self.crime_data_path = crime_data_path
        self.conviction_data_path = conviction_data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.crime_data = None
        self.conviction_data = None
        self.model = None
        self.scaler = None
        self.encoder = None

    def load_and_preprocess_data(self):
        """Load and preprocess crime and conviction data."""
        try:
            # Load crime data
            self.crime_data = pd.read_csv(self.crime_data_path)
            print("Crime data loaded successfully.")
            print("Columns in crime_data:", self.crime_data.columns)

            # Load conviction data
            self.conviction_data = pd.read_csv(self.conviction_data_path)
            print("Conviction data loaded successfully.")

            # Preprocess crime data
            self.crime_data['Date'] = pd.to_datetime(self.crime_data['Date'], format='%Y-%m-%d', errors='coerce')

            # Ensure latitude and longitude are separate columns
            self.crime_data['Latitude'] = self.crime_data['Latitude'].astype(str).str.strip()
            lat_long_split = self.crime_data['Latitude'].str.split(',', expand=True)

            # Handle missing or incorrect values
            if lat_long_split.shape[1] == 2:
                self.crime_data[['Latitude', 'Longitude']] = lat_long_split
            else:
                self.crime_data['Latitude'], self.crime_data['Longitude'] = None, None

            # Convert latitude and longitude to numeric values
            self.crime_data['Latitude'] = pd.to_numeric(self.crime_data['Latitude'], errors='coerce')
            self.crime_data['Longitude'] = pd.to_numeric(self.crime_data['Longitude'], errors='coerce')

            # Preprocess conviction data
            self.conviction_data['Date_Filed'] = pd.to_datetime(self.conviction_data['Date_Filed'])
            self.conviction_data['Date_Resolved'] = pd.to_datetime(self.conviction_data['Date_Resolved'])
            self.conviction_data['Evidence_Quality'] = self.conviction_data['Evidence_Quality'].map({'Low': 0, 'Medium': 1, 'High': 2})

            print("Data preprocessing completed.")
        except Exception as e:
            print(f"Error during data loading/preprocessing: {e}")
            raise

    def load_model(self):
        """Load the pre-trained crime prediction model, scaler, and encoder."""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.encoder = joblib.load(self.encoder_path)
            print("Crime prediction model, scaler, and encoder loaded successfully.")
        except Exception as e:
            print(f"Error loading model, scaler, or encoder: {e}")
            raise

    def calculate_conviction_rate(self):
        """Calculate historical conviction rates."""
        try:
            # Group by district and calculate conviction rate
            conviction_rate = self.conviction_data.groupby('District').apply(
                lambda x: (x['Conviction_Status'].sum() / len(x)) * 100).reset_index(name='Conviction_Rate')
            conviction_rate.to_csv("historical_conviction_rates.csv", index=False)
            print("Historical conviction rates saved to 'historical_conviction_rates.csv'.")
            return conviction_rate
        except Exception as e:
            print(f"Error during conviction rate calculation: {e}")
            raise

    def predict(self, data):
        """Predict crime probability for given data."""
        try:
            if self.model is None or self.scaler is None or self.encoder is None:
                raise ValueError("Model, scaler, or encoder not loaded. Call load_model() first.")

            # Normalize district names
            data['District'] = data['District'].str.strip().str.title()
            self.crime_data['District'] = self.crime_data['District'].str.strip().str.title()

            district = data['District'].iloc[0]
            district_data = self.crime_data[self.crime_data['District'] == district]

            if district_data.empty:
                raise ValueError(f"No data found for district: {district}")

            # Ensure required features exist
            required_features = ['Latitude', 'Longitude', 'Population Density (per sq. km)', 'Average Income (INR)', 'Unemployment Rate (%)']
            missing_features = [col for col in required_features if col not in data.columns]

            if missing_features:
                print(f"Warning: Missing features in input data - {missing_features}")
                for col in missing_features:
                    data[col] = self.crime_data[col].mean()  # Fill missing columns with dataset mean

            # Assign correct latitude and longitude
            data['Latitude'] = district_data['Latitude'].iloc[0]
            data['Longitude'] = district_data['Longitude'].iloc[0]

            # Prepare the data for prediction
            X = data[required_features].fillna(data[required_features].mean())
            X_scaled = self.scaler.transform(X)  # Scale the features

            # Make predictions
            predicted_probabilities = self.model.predict_proba(X_scaled)
            selected_crime_type = data['Crime_Type'].iloc[0]
            selected_crime_index = np.where(self.encoder.categories_[0] == selected_crime_type)[0][0]
            crime_probability = predicted_probabilities[0][selected_crime_index]

            return crime_probability

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    

    def get_historical_crimes_column(self):
        """Find the correct column name for historical crimes dynamically."""
        possible_names = [col for col in self.crime_data.columns if "historical" in col.lower()]
        return possible_names[0] if possible_names else None
