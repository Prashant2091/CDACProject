import streamlit as st
import numpy as np
import pandas as pd
import requests
import datetime
from scipy.spatial.distance import cityblock
import pickle
import pydeck as pdk
from streamlit_geolocation import streamlit_geolocation

# Load trained model
model = pickle.load(open("model1.pkl", "rb"))

# Google Geocoding API function
def get_location_by_address(address, api_key):
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
            st.warning(f"Geocoding Error: {data['status']} for '{address}'")
            return None, None, None
    except Exception as e:
        st.error(f"Geocoding Connection error: {e}")
        return None, None, None

# Weather retrieval by precise coordinates
def weather_by_coordinates(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={api_key.strip()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['main']['temp'], data['weather'][0]['description'].title()
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return None, None

# Weather factor determination
def determine_weather_factor(condition):
    condition = condition.lower()
    factors = {"rain":1.2, "snow":1.3, "thunderstorm":1.4, 
               "fog":1.15, "mist":1.15, "cloud":1.1, "clear":1.0}
    return next((factors[key] for key in factors if key in condition), 1.05)

# Adaptive zoom
def adaptive_zoom(p_lat, p_lon, d_lat, d_lon):
    distance = cityblock((p_lat, p_lon), (d_lat, d_lon))
    if distance < 2: return 14
    elif distance < 10: return 13
    elif distance < 25: return 12
    elif distance < 50: return 11
    elif distance < 100: return 10
    return 8

# Streamlit UI
st.set_page_config(layout="wide")
st.image("uber.jpg")
st.title("🚖 Uber Ride Price Prediction")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("📅 Pickup Date")
    time = st.time_input("⏰ Pickup Time", datetime.datetime.now().time())
with col2:
    passenger_count = st.selectbox("👥 Passengers", range(1, 7))

# Live location detection
use_live_location = st.checkbox("📍 Use my current location as pickup")
if use_live_location:
    loc = streamlit_geolocation()
    if loc:
        p_lat, p_lon = loc['latitude'], loc['longitude']
        st.success(f"Live location detected: {p_lat:.6f}, {p_lon:.6f}")
        p_formatted = "Live Location"
    else:
        st.info("Awaiting location...")
else:
    pickup_address = st.text_input("📍 Pickup Location")
    p_lat, p_lon, p_formatted = get_location_by_address(pickup_address, api_key="AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")

dropoff_address = st.text_input("📍 Dropoff Location")
d_lat, d_lon, d_formatted = get_location_by_address(dropoff_address, api_key="AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")

# Weather data
pickup_temp, pickup_cond = weather_by_coordinates(p_lat, p_lon, "665b90b40a24cf1e5d00fb6055c5b757")
dropoff_temp, dropoff_cond = weather_by_coordinates(d_lat, d_lon, "665b90b40a24cf1e5d00fb6055c5b757")
st.info(f"Pickup: {pickup_temp}°F | {pickup_cond}")
st.info(f"Dropoff: {dropoff_temp}°F | {dropoff_cond}")

# Dynamic map with adaptive zoom
if None not in [p_lat, p_lon, d_lat, d_lon]:
    map_data = pd.DataFrame({
        'lat': [p_lat, d_lat],
        'lon': [p_lon, d_lon],
        'location': ['Pickup', 'Dropoff'],
        'address': [p_formatted, d_formatted],
        'weather': [pickup_cond, dropoff_cond]
    })
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=np.mean([p_lat, d_lat]),
            longitude=np.mean([p_lon, d_lon]),
            zoom=adaptive_zoom(p_lat, p_lon, d_lat, d_lon), pitch=45),
        layers=[
            pdk.Layer('ScatterplotLayer', data=map_data, get_position='[lon, lat]', get_color='[0,128,255]', get_radius=200, pickable=True),
            pdk.Layer('TextLayer', data=map_data, get_position='[lon, lat]', get_text='location', get_color='[0,0,0]', get_size=16)],
        tooltip={"text": "{location}: {address} | Weather: {weather}"}
    ))

# Fare prediction with adaptive temp weighting
if st.button("💲 Predict Fare"):
    if None in [pickup_temp, dropoff_temp]: st.error("Missing weather data.")
    else:
        dist = cityblock((p_lat, p_lon), (d_lat, d_lon))
        pickup_weight = cityblock((p_lat, p_lon), ((p_lat+d_lat)/2, (p_lon+d_lon)/2)) / dist if dist else 0.5
        temp = pickup_temp*(1-pickup_weight) + dropoff_temp*pickup_weight
        pickup_factor, dropoff_factor = determine_weather_factor(pickup_cond), determine_weather_factor(dropoff_cond)
        weather_factor = pickup_factor*(1-pickup_weight) + dropoff_factor*pickup_weight
        features = np.array([
            p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour, date.day, date.month, date.year,
            cityblock((40.7141667,-74.0063889),(p_lat,p_lon)),
            cityblock((40.6441667,-73.7822222),(p_lat,p_lon)),
            cityblock((40.6441667,-73.7822222),(d_lat,d_lon)),
            cityblock((40.69,-74.175),(p_lat,p_lon)),
            cityblock((40.69,-74.175),(d_lat,d_lon)),
            cityblock((40.77,-73.87),(p_lat,p_lon)),
            cityblock((40.77,-73.87),(d_lat,d_lon)),
            d_lon-p_lon, d_lat-p_lat, dist, temp
        ]).reshape(1,-1)
        fare = abs(model.predict(features)[0])*(1+0.1*(passenger_count-1))*weather_factor
        st.success(f"✅ Predicted Fare: ${fare:.2f}")

