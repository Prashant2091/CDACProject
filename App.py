import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cityblock
import pickle
import datetime

# API compliance headers
geo_headers = {
    'User-Agent': 'UberPriceApp/1.0 (prashantjack.shukla@gmail.com)'
}

# Load your trained model
model = pickle.load(open("model1.pkl", "rb"))

# Robust geolocation function using OpenStreetMap Nominatim API
def get_location_by_address(address, api_key):
    if not address.strip():
        st.warning("Address field is empty.")
        return None, None
    try:
        url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            st.warning(f"Google Geocoding Error: {data['status']} for '{address}'")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None, None

# Dynamic Weather function using OpenWeatherMap API
def weather(city, api_key):
    try:
        city_query = city.strip().replace(" ", "+")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_query}&units=imperial&appid={api_key.strip()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        location = f"{weather_data.get('name', 'Unknown')}, {weather_data['sys'].get('country', '')}"
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description'].title()

        st.success(f"✅ Location: {location}")
        st.info(f"🌡️ Temperature: {temperature} °F | ☁️ Condition: {condition}")

        return temperature

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e}")
    except Exception as e:
        st.error(f"Error retrieving weather: {e}")
    return None

# Streamlit app interface
st.image("uber.jpg")
st.title("🚖 Uber Ride Price Prediction")

st.markdown("### 🌎 Enter Your City Name")
#st.markdown("_Use format: **City,Country_Code** (e.g., Jhansi,IN, Paris,FR)_")
city = st.text_input("City", "New York,US")
temperature = weather(city, api_key="665b90b40a24cf1e5d00fb6055c5b757")

# Date and Time Input
date = st.date_input("📅 Pickup Date")
time = st.time_input("⏰ Pickup Time", datetime.time(0, 0))
st.info(f"Pickup Date & Time: {date} at {time}")

# Passenger Count
passenger_count = st.selectbox("👥 Passengers", np.arange(1, 7))
st.info(f"Passengers: {passenger_count}")

# Pickup Location
pickup_address = st.text_input("📍 Pickup Location")
p_lat, p_lon = get_location_by_address(pickup_address, api_key="AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")
if p_lat and p_lon:
    st.success(f"Pickup Coordinates: {p_lat}, {p_lon}")
else:
    st.warning("Enter pickup coordinates manually.")

p_lat = st.number_input("Pickup Latitude", value=p_lat or 0.0, format="%.6f")
p_lon = st.number_input("Pickup Longitude", value=p_lon or 0.0, format="%.6f")

# Dropoff Location
dropoff_address = st.text_input("📍 Dropoff Location")
d_lat, d_lon = get_location_by_address(dropoff_address, api_key="AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")
if d_lat and d_lon:
    st.success(f"Dropoff Coordinates: {d_lat}, {d_lon}")
else:
    st.warning("Enter dropoff coordinates manually.")

d_lat = st.number_input("Dropoff Latitude", value=d_lat or 0.0, format="%.6f")
d_lon = st.number_input("Dropoff Longitude", value=d_lon or 0.0, format="%.6f")

# Display Map
map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
st.map(map_data)

# Fare Prediction
if st.button("💲 Predict Fare"):
    if temperature is None:
        st.error("Weather data unavailable.")
    else:
        features = np.array([
            p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour,
            date.day, date.month, date.year,
            cityblock((40.7141667, -74.0063889), (p_lat, p_lon)),
            cityblock((40.6441666667, -73.7822222222), (p_lat, p_lon)),
            cityblock((40.6441666667, -73.7822222222), (d_lat, d_lon)),
            cityblock((40.69, -74.175), (p_lat, p_lon)),
            cityblock((40.69, -74.175), (d_lat, d_lon)),
            cityblock((40.77, -73.87), (p_lat, p_lon)),
            cityblock((40.77, -73.87), (d_lat, d_lon)),
            d_lon - p_lon, d_lat - p_lat,
            cityblock((p_lat, p_lon), (d_lat, d_lon)), temperature
        ]).reshape(1, -1)

        fare = model.predict(features)
        st.success(f"✅ Predicted Fare: ${abs(fare[0]):.2f}")
