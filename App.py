import streamlit as st
import numpy as np
import pandas as pd
import requests
import datetime
from scipy.spatial.distance import cityblock
import pickle
import pydeck as pdk

# API compliance headers
geo_headers = {
    'User-Agent': 'UberPriceApp/1.0 (prashantjack.shukla@gmail.com)'
}

# Load your trained model
model = pickle.load(open("model1.pkl", "rb"))

# Geolocation function using Google Geocoding API
def get_location_by_address(address, api_key):
    if not address.strip():
        st.warning("Address field is empty.")
        return None, None, None
    try:
        url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            formatted_address = data['results'][0]['formatted_address']
            return location['lat'], location['lng'], formatted_address
        else:
            st.warning(f"Google Geocoding Error: {data['status']} for '{address}'")
            return None, None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None, None, None

# Weather retrieval function using OpenWeatherMap API
def weather(city, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=imperial&appid={api_key.strip()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        location = f"{weather_data.get('name', 'Unknown')}, {weather_data['sys'].get('country', '')}"
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description'].title()
        st.success(f"✅ Location: {location}")
        st.info(f"🌡️ Temperature: {temperature} °F | ☁️ Condition: {condition}")
        return temperature, condition
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return None, None

# Weather-based fare adjustment
def determine_weather_factor(condition):
    condition = condition.lower()
    if "rain" in condition:
        return 1.20
    elif "snow" in condition:
        return 1.30
    elif "thunderstorm" in condition:
        return 1.40
    elif "fog" in condition or "mist" in condition:
        return 1.15
    elif "cloud" in condition:
        return 1.10
    elif "clear" in condition:
        return 1.0
    else:
        return 1.05

# Streamlit UI
st.set_page_config(page_title="Uber Ride Price Prediction", layout="wide")
st.image("uber.jpg")
st.title("🚖 Uber Ride Price Prediction")

city = st.text_input("🌎 Enter Your City (City,Country)", "New York,US")
temperature, condition = weather(city, api_key="665b90b40a24cf1e5d00fb6055c5b757")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("📅 Pickup Date")
    time = st.time_input("⏰ Pickup Time", datetime.time(0, 0))

with col2:
    passenger_count = st.selectbox("👥 Number of Passengers", range(1, 7))

pickup_address = st.text_input("📍 Pickup Location")
p_lat, p_lon, p_formatted = get_location_by_address(pickup_address, api_key="AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")

p_lat = st.number_input("Pickup Latitude", value=p_lat or 0.0, format="%.6f")
p_lon = st.number_input("Pickup Longitude", value=p_lon or 0.0, format="%.6f")

dropoff_address = st.text_input("📍 Dropoff Location")
d_lat, d_lon, d_formatted = get_location_by_address(dropoff_address, api_key="AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")

d_lat = st.number_input("Dropoff Latitude", value=d_lat or 0.0, format="%.6f")
d_lon = st.number_input("Dropoff Longitude", value=d_lon or 0.0, format="%.6f")

map_data = pd.DataFrame({
    'lat': [p_lat, d_lat],
    'lon': [p_lon, d_lon],
    'location': ['Pickup', 'Dropoff'],
    'address': [p_formatted or "Pickup Location", d_formatted or "Dropoff Location"]
})

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/streets-v11',
    initial_view_state=pdk.ViewState(
        latitude=(p_lat + d_lat) / 2,
        longitude=(p_lon + d_lon) / 2,
        zoom=12,
        pitch=45
    ),
    layers=[
        pdk.Layer('ScatterplotLayer',
                  data=map_data,
                  get_position='[lon, lat]',
                  get_color='[255, 0, 0]',
                  get_radius=200),
        pdk.Layer('TextLayer',
                  data=map_data,
                  get_position='[lon, lat]',
                  get_text='location',
                  get_color='[0,0,0]',
                  get_size=16)
    ],
    tooltip={"text": "{location}: {address}"}
))

if st.button("💲 Predict Fare"):
    if None in [temperature, condition]:
        st.error("Missing data, please check inputs.")
    else:
        weather_factor = determine_weather_factor(condition)
        features = np.array([
            p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour,
            date.day, date.month, date.year,
            cityblock((40.7141667, -74.0063889), (p_lat, p_lon)),
            cityblock((40.6441667, -73.7822222), (p_lat, p_lon)),
            cityblock((40.6441667, -73.7822222), (d_lat, d_lon)),
            cityblock((40.69, -74.175), (p_lat, p_lon)),
            cityblock((40.69, -74.175), (d_lat, d_lon)),
            cityblock((40.77, -73.87), (p_lat, p_lon)),
            cityblock((40.77, -73.87), (d_lat, d_lon)),
            d_lon - p_lon, d_lat - p_lat,
            cityblock((p_lat, p_lon), (d_lat, d_lon)),
            temperature
        ]).reshape(1, -1)
        base_fare = model.predict(features)[0]
        fare = abs(base_fare) * (1 + 0.1*(passenger_count-1)) * weather_factor
        st.success(f"✅ Predicted Fare: ${fare:.2f}")

st.markdown("[Book Your Ride on Uber](https://www.uber.com)")
