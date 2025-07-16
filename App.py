import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.spatial.distance import cityblock
import pickle 
import datetime

# Required headers for Nominatim API compliance
geo_headers = {
    'User-Agent': 'UberPriceApp/1.0 (prashantjack.shukla@gmail.com)'
}

# Load your trained model
model = pickle.load(open("model1.pkl", "rb"))

# Get latitude and longitude from OpenStreetMap Nominatim
def get_location_by_address(address):
    if not address.strip():
        st.warning("Address field is empty.")
        return None, None
    try:
        url = f'https://nominatim.openstreetmap.org/search?q={address}&format=json'
        response = requests.get(url, headers=geo_headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            st.warning(f"No coordinates found for '{address}'.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None, None

# Get weather data using OpenWeatherMap API
def weather(city, api_key):
    if not api_key or "your_valid_api_key" in api_key:
        st.error("Please set a valid OpenWeatherMap API key.")
        return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=imperial&appid={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        location = weather_data.get('name', 'Unknown Location')
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description'].title()

        st.write(f"Location: {location}")
        st.write(f"Temperature: {temperature} °F")
        st.write(f"Condition: {condition}")

        return temperature
    except requests.exceptions.RequestException as e:
        st.error(f"Weather API error: {e}")
        return None

# Streamlit app interface
st.image("uber.jpg")
st.title("Uber Ride Price Prediction Using Multiple Factors")

# Live Weather
city = st.text_input("Enter Your City Name:", "New York")
temperature = weather(city, api_key="665b90b40a24cf1e5d00fb6055c5b757")  # Replace with your actual API key

# Date and Time Input
date = st.date_input("Date of Pickup")
time = st.time_input("Enter Pickup Time", datetime.time(0, 0))

st.info(f"Pickup Date: {date} at {time}")

# Passenger Count
passenger_count = st.selectbox("Number of Passengers", np.arange(1, 7))
st.info(f"Passengers: {passenger_count}")

# Pickup Location
street = st.text_input("Pickup Location")
p_lat, p_lon = get_location_by_address(street)
if p_lat and p_lon:
    st.success(f"Pickup Coordinates: {p_lat}, {p_lon}")
else:
    st.warning("Enter pickup coordinates manually.")

p_lat = st.number_input("Pickup Latitude", value=p_lat or 0.0, format="%.6f")
p_lon = st.number_input("Pickup Longitude", value=p_lon or 0.0, format="%.6f")

# Dropoff Location
street1 = st.text_input("Dropoff Location")
d_lat, d_lon = get_location_by_address(street1)
if d_lat and d_lon:
    st.success(f"Dropoff Coordinates: {d_lat}, {d_lon}")
else:
    st.warning("Enter dropoff coordinates manually.")

d_lat = st.number_input("Dropoff Latitude", value=d_lat or 0.0, format="%.6f")
d_lon = st.number_input("Dropoff Longitude", value=d_lon or 0.0, format="%.6f")

# Display Map
map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
st.map(map_data)

# Fare Prediction Logic
if st.button("Predict Fare"):
    if temperature is None:
        st.error("Weather data unavailable. Please resolve before predicting.")
    else:
        dist_to_cent = cityblock((40.7141667, -74.0063889), (p_lat, p_lon))
        pick_dist_to_jfk = cityblock((40.6441666667, -73.7822222222), (p_lat, p_lon))
        drop_dist_to_jfk = cityblock((40.6441666667, -73.7822222222), (d_lat, d_lon))
        pick_dist_to_ewr = cityblock((40.69, -74.175), (p_lat, p_lon))
        drop_dist_to_ewr = cityblock((40.69, -74.175), (d_lat, d_lon))
        pick_dist_to_lgr = cityblock((40.77, -73.87), (p_lat, p_lon))
        drop_dist_to_lgr = cityblock((40.77, -73.87), (d_lat, d_lon))
        long_diff = d_lon - p_lon
        lat_diff = d_lat - p_lat
        manhattan_dist = cityblock((p_lat, p_lon), (d_lat, d_lon))

        features = np.array([p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour, date.day, date.month, date.year,
                             dist_to_cent, pick_dist_to_jfk, drop_dist_to_jfk, pick_dist_to_ewr, drop_dist_to_ewr,
                             pick_dist_to_lgr, drop_dist_to_lgr, long_diff, lat_diff, manhattan_dist, temperature]).reshape(1, -1)

        fare = model.predict(features)
        st.success(f"✅ The Predicted Fare is: ${abs(fare[0]):.2f}")

# Sidebar Information
df = pd.DataFrame({
    "Name": ["Dixit Dutt Bohra", "Lalit Bhaskar Mahale", "Manchikatla Raman Kumar", "Prashant Shukla", "Vipin Kumar Tripathi"],
    "PRN": ["220340128010", "220340128021", "220340128023", "220340128036", "220340128054"],
    "Email": ["dixitduttbohra@gmail.com", "lalitmahale121@gmail.com", "ramanmenche@gmail.com", "prashantjack.shukla@gmail.com", "tripathivipin078@gmail.com"]
}, index=np.arange(1, 6))

sidebar = st.sidebar.selectbox("Uber Ride Price Prediction", ["", "Developers", "Guide"])

if sidebar == "Developers":
    st.sidebar.image("developer.jpg")
    st.sidebar.table(df)
elif sidebar == "Guide":
    st.sidebar.header("Guide: Pramod Kumar Sharma")
    st.sidebar.image("pramod_sir.jpg")
    st.sidebar.image("cdac_pune.jpg")
