# frontend/app.py
import gradio as gr
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os
import http.server
import socketserver
import threading
import numpy as np  # For handling numeric data
from geopy.geocoders import Nominatim
from prediction import CrimeAnalysis  # Updated import path

# Load the CSV data ONCE at startup
crime_data = pd.read_csv("DATASET/Crime_Prediction.csv")
odisha_data = pd.read_csv("DATASET/Odisha_Crime_Data.csv")

# Handle missing 'Number_of_Cases' column
if 'Number_of_Cases' not in odisha_data.columns:
    print("Warning: 'Number_of_Cases' column missing! Creating default values...")
    odisha_data['Number_of_Cases'] = np.random.randint(1, 50, size=len(odisha_data))  # Placeholder values

# Geolocation caching
geolocator = Nominatim(user_agent="geoapi")
geocode_cache = {}

def get_block_coordinates(block):
    if block in geocode_cache:
        return geocode_cache[block]
    try:
        location = geolocator.geocode(f"{block}, Odisha, India")
        if location:
            geocode_cache[block] = [location.latitude, location.longitude]
            return geocode_cache[block]
    except:
        pass
    return [None, None]

odisha_data[['Latitude', 'Longitude']] = odisha_data['Block'].apply(lambda x: pd.Series(get_block_coordinates(x)))

# Extract unique district names and dates
districts = crime_data['District'].unique().tolist()
dates = crime_data['Date'].unique().tolist()

# Start a lightweight HTTP server for local hosting
PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler
server_thread = threading.Thread(target=lambda: socketserver.TCPServer(("", PORT), Handler).serve_forever(), daemon=True)
server_thread.start()

def predict_crime(district, crime_type, date, analyzer):
    """Predict crime probability and generate heatmap."""
    new_data = pd.DataFrame({'District': [district], 'Crime_Type': [crime_type], 'Date': [date]})

    try:
        new_data['Date'] = pd.to_datetime(new_data['Date']).astype(int) / 10**9
        crime_probability = analyzer.predict(new_data)
        conviction_rate = analyzer.calculate_conviction_rate()

        # Get conviction rate
        district_conviction_rate = conviction_rate.loc[conviction_rate['District'] == district, 'Conviction_Rate'].values
        district_conviction_rate = float(district_conviction_rate[0]) if len(district_conviction_rate) > 0 else "N/A"

        # Create map
        crime_map = folium.Map(location=[odisha_data['Latitude'].mean(), odisha_data['Longitude'].mean()], zoom_start=10)
        heat_data = odisha_data[(odisha_data['District'] == district) & (odisha_data['Crime_Type'] == crime_type)]

        if not heat_data.empty:
            heat_data = heat_data[['Latitude', 'Longitude', 'Number_of_Cases']].dropna()
            max_crime_rate = heat_data['Number_of_Cases'].max()
            if max_crime_rate > 0:
                heat_data['Normalized_Crime'] = heat_data['Number_of_Cases'] / max_crime_rate

            HeatMap(heat_data[['Latitude', 'Longitude', 'Normalized_Crime']].values.tolist(), radius=25, blur=15).add_to(crime_map)
        else:
            folium.Marker([odisha_data['Latitude'].mean(), odisha_data['Longitude'].mean()], popup="No data").add_to(crime_map)

        crime_map.save("crime_map.html")
        return f"{crime_probability:.2f}", f"{district_conviction_rate:.2f}", f"http://localhost:{PORT}/crime_map.html", f'<iframe src="http://localhost:{PORT}/crime_map.html" width="100%" height="400"></iframe>'

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: {e}", None, None, None

def launch_gradio_app(analyzer):
    with gr.Blocks() as demo:
        gr.Markdown("# Crime Prediction App")
        with gr.Row():
            with gr.Column():
                district = gr.Dropdown(label="District", choices=districts, value=districts[0])
                crime_type = gr.Dropdown(label="Crime Type", choices=["Theft", "Robbery", "Murder"], value="Theft")
                date = gr.Dropdown(label="Date", choices=dates, value=dates[0])
                predict_button = gr.Button("Predict Crime")
            with gr.Column():
                crime_probability = gr.Textbox(label="Crime Probability (%)")
                conviction_rate = gr.Textbox(label="Conviction Rate (%)")
                map_url_output = gr.Textbox(label="Map URL")
                map_output = gr.HTML(label="Crime Map")
        predict_button.click(predict_crime, inputs=[district, crime_type, date, gr.State(analyzer)], outputs=[crime_probability, conviction_rate, map_url_output, map_output])
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    analyzer = CrimeAnalysis()
    analyzer.load_model()
    launch_gradio_app(analyzer)
