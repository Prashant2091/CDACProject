import streamlit as st
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
from scipy.spatial.distance import cityblock
import pickle 
import datetime

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

model = pickle.load(open("model1.pkl", "rb"))

def get_location_by_address(address):
    try:
        url = f'https://nominatim.openstreetmap.org/search?q={address}&format=json'
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return float(lat), float(lon)
        else:
            st.warning(f"No location found for '{address}'. Please verify the address.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error while fetching coordinates: {e}")
        return None, None

def weather(city, api_key="weather_api_lattest"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=imperial&appid={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        location = weather_data['name']
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description'].title()

        st.write(f"Location: {location}")
        st.write(f"Temperature: {temperature} °F")
        st.write(f"Condition: {condition}")

        return temperature
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving weather: {e}")
        return None


# Title and Logo 
st.image("uber.jpg")
st.title("Uber Ride Price Prediction Using Multiple Factors")

# Live Weather 
city = st.text_input("Enter Your City Name:", "New York")
temperature = weather(city)

# Datetime 
date = st.date_input("Date of Pickup")
st.info(date)

# Passenger Count
passenger_count = st.selectbox("Passenger", np.arange(1, 7))
st.info(passenger_count)

time = st.time_input("Enter Pickup Time", datetime.time(0, 0))
st.info(time)

# Pickup Location
street = st.text_input("Pickup Location")
lat, lon = get_location_by_address(street)
if lat and lon:
    st.write(f"Latitude: {lat}, Longitude: {lon}")
else:
    st.write("Could not find coordinates for the pickup location.")
p_lat = st.number_input("Enter Pickup Latitude", step=1e-6, format="%.6f")
p_lon = st.number_input("Enter Pickup Longitude", step=1e-6, format="%.6f")

# Dropoff Location
street1 = st.text_input("Dropoff Location")
lat1, lon1 = get_location_by_address(street1)
if lat1 and lon1:
    st.write(f"Latitude: {lat1}, Longitude: {lon1}")
else:
    st.write("Could not find coordinates for the dropoff location.")
d_lat = st.number_input("Enter Dropoff Latitude", step=1e-6, format="%.6f")
d_lon = st.number_input("Enter Dropoff Longitude", step=1e-6, format="%.6f")

# Map visualization
map_data1 = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
st.map(map_data1)

if st.button("Predict Fare"):
    day, month, year = date.day, date.month, date.year
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

    prediction = np.array([p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour, day, month, year,
                          dist_to_cent, pick_dist_to_jfk, drop_dist_to_jfk,
                          pick_dist_to_ewr, drop_dist_to_ewr, pick_dist_to_lgr, drop_dist_to_lgr,
                          long_diff, lat_diff, manhattan_dist, temperature]).reshape(1, -1)
    
    result = model.predict(prediction)
    st.write("The Predicted Fare is: $", abs(result[0]))

# Sidebar
data = {"Name": ["Dixit Dutt Bohra", "Lalit Bhaskar Mahale", "Manchikatla Raman Kumar", "Prashant Shukla", "Vipin Kumar Tripathi"],
        "PRN ": ["220340128010", "220340128021", "220340128023", "220340128036", "220340128054"],
        "Email ": ["dixitduttbohra@gmail.com", "lalitmahale121@gmail.com", "ramanmenche@gmail.com", "prashantjack.shukla@gmail.com", "tripathivipin078@gmail.com"]}
df = pd.DataFrame(data, index=np.arange(1, 6))

sidebar = st.sidebar.selectbox("Uber Ride Price Prediction", ["", "Developers", "Guide"])

if sidebar == "Developers":
    st.image("developer.jpg")
    st.table(df)
elif sidebar == "Guide":
    st.header("Guide")
    st.image("pramod_sir.jpg")
    st.header("Pramod Kumar Sharma")
    st.image("cdac_pune.jpg")
