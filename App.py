'''import streamlit as st
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

# Precise Weather retrieval function using OpenWeatherMap API with coordinates
def weather_by_coordinates(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={api_key.strip()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description'].title()
        return temperature, condition
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return None, None


# Determine weather factor
def weather_factor(condition):
    condition = condition.lower()
    factors = {"rain":1.2, "snow":1.3, "thunderstorm":1.4,
               "fog":1.15, "mist":1.15, "cloud":1.1, "clear":1.0}
    return next((factors[key] for key in factors if key in condition), 1.05)

# Adaptive zoom based on distance
def adaptive_zoom(distance):
    return 14 if distance<2 else 13 if distance<10 else 12 if distance<25 else 11 if distance<50 else 10 if distance<100 else 8

# Haversine distance
from geopy.distance import geodesic
def get_distance(p1, p2):
    return geodesic(p1, p2).miles

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
use_live = st.checkbox("📍 Use Current Location as Pickup")
if use_live:
    loc = streamlit_geolocation()
    if loc:
        p_lat, p_lon = loc['latitude'], loc['longitude']
        p_formatted = "Live Location"
        st.success(f"Detected Live Location: {p_lat:.6f}, {p_lon:.6f}")
    else:
        st.info("Awaiting location permission...")
else:
    pickup_address = st.text_input("📍 Pickup Location")
if pickup_address.strip():
    p_lat, p_lon, p_formatted = get_location_by_address(pickup_address, "AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")
else:
    p_lat = p_lon = p_formatted = None
    st.warning("Please enter a valid pickup location.")
p_lat = st.number_input("Pickup Latitude", value=p_lat or 0.0, format="%.6f")
p_lon = st.number_input("Pickup Longitude", value=p_lon or 0.0, format="%.6f")
dropoff_address = st.text_input("📍 Dropoff Location")
if dropoff_address.strip():
    d_lat, d_lon, d_formatted = get_location_by_address(dropoff_address, "AIzaSyCapre4-pQ70FiV5EPMpIvs7TPbFzU1bAQ")
else:
    d_lat = d_lon = d_formatted = None
    st.warning("Please enter a valid dropoff location.")
d_lat = st.number_input("Dropoff Latitude", value=d_lat or 0.0, format="%.6f")
d_lon = st.number_input("Dropoff Longitude", value=d_lon or 0.0, format="%.6f")



# Fetch weather for pickup/dropoff
pickup_temp, pickup_cond = weather_by_coordinates(p_lat, p_lon, "665b90b40a24cf1e5d00fb6055c5b757")
dropoff_temp, dropoff_cond = weather_by_coordinates(d_lat, d_lon, "665b90b40a24cf1e5d00fb6055c5b757")
st.info(f"Pickup Weather: {pickup_temp}°F, {pickup_cond}")
st.info(f"Dropoff Weather: {dropoff_temp}°F, {dropoff_cond}")

# Calculate actual distance
if None not in [p_lat, p_lon, d_lat, d_lon]:
    actual_distance = get_distance((p_lat, p_lon), (d_lat, d_lon))
    st.success(f"🗺️ Trip Distance: {actual_distance:.2f} miles")

    map_data = pd.DataFrame({
        'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon],
        'location': ['Pickup', 'Dropoff'],
        'address': [p_formatted, d_formatted],
        'weather': [pickup_cond, dropoff_cond]
    })

    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=np.mean([p_lat, d_lat]),
            longitude=np.mean([p_lon, d_lon]),
            zoom=adaptive_zoom(actual_distance), pitch=45),
        layers=[
            pdk.Layer('ScatterplotLayer', data=map_data, get_position='[lon, lat]',
                      get_color='[255,0,0]', get_radius=200, pickable=True),
            pdk.Layer('TextLayer', data=map_data, get_position='[lon, lat]',
                      get_text='location', get_color='[0,0,0]', get_size=16)],
        tooltip={"text": "{location}: {address}\nWeather: {weather}"}
    ))

if st.button("💲 Predict Fare"):
    if None in [pickup_temp, dropoff_temp]:
        st.error("Incomplete weather data.")
    else:
        # Adaptive weighted temperature & weather
        pickup_weight = 0.5 if actual_distance == 0 else min(1, 5/actual_distance)
        avg_temp = pickup_temp * pickup_weight + dropoff_temp * (1-pickup_weight)
        weather_adj = weather_factor(pickup_cond)*pickup_weight + weather_factor(dropoff_cond)*(1-pickup_weight)

        # Features array
        features = np.array([
            p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour,
            date.day, date.month, date.year,
            cityblock((40.7141667,-74.0063889),(p_lat,p_lon)),
            cityblock((40.6441667,-73.7822222),(p_lat,p_lon)),
            cityblock((40.6441667,-73.7822222),(d_lat,d_lon)),
            cityblock((40.69,-74.175),(p_lat,p_lon)),
            cityblock((40.69,-74.175),(d_lat,d_lon)),
            cityblock((40.77,-73.87),(p_lat,p_lon)),
            cityblock((40.77,-73.87),(d_lat,d_lon)),
            d_lon-p_lon, d_lat-p_lat, actual_distance, avg_temp
        ]).reshape(1,-1)

        base_fare = abs(model.predict(features)[0])
        final_fare = base_fare*(1+0.1*(passenger_count-1))*weather_adj
        st.success(f"✅ Predicted Fare: ${final_fare:.2f}")'''

import streamlit as st
import numpy as np
import pandas as pd
import requests
import datetime
from scipy.spatial.distance import cityblock
import pickle
import pydeck as pdk
from streamlit_geolocation import streamlit_geolocation

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

# Precise Weather retrieval function using OpenWeatherMap API with coordinates
def weather_by_coordinates(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=imperial&appid={api_key.strip()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description'].title()
        return temperature, condition
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return None, None

# Comprehensive weather-based fare adjustment
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
# Haversine distance
from geopy.distance import geodesic
def get_distance(p1, p2):
    return geodesic(p1, p2).miles
# Streamlit UI
st.set_page_config(page_title="Uber Ride Price Prediction", layout="wide")
st.image("uber.jpg")
st.title("🚖 Uber Ride Price Prediction")

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

# Fetching precise weather for both pickup and dropoff locations
pickup_temp, pickup_condition = weather_by_coordinates(p_lat, p_lon, api_key="665b90b40a24cf1e5d00fb6055c5b757")
dropoff_temp, dropoff_condition = weather_by_coordinates(d_lat, d_lon, api_key="665b90b40a24cf1e5d00fb6055c5b757")

st.info(f"Pickup Weather: {pickup_temp}°F | {pickup_condition}")
st.info(f"Dropoff Weather: {dropoff_temp}°F | {dropoff_condition}")

# Display Map
map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
st.map(map_data)
pickup_weather_factor = determine_weather_factor(pickup_condition)
dropoff_weather_factor = determine_weather_factor(dropoff_condition)

# Average factor for better accuracy
final_weather_factor = (pickup_weather_factor + dropoff_weather_factor) / 2
if st.button("💲 Predict Fare"):
    if None in [pickup_temp, dropoff_temp]:
        st.error("Incomplete weather data.")
    else:
        # Recalculate original features precisely
        dist_to_cent = cityblock((40.7141667,-74.0063889), (p_lat,p_lon))
        pickup_dist_jfk = cityblock((40.6441667,-73.7822222), (p_lat,p_lon))
        dropoff_dist_jfk = cityblock((40.6441667,-73.7822222), (d_lat,d_lon))
        pickup_dist_ewr = cityblock((40.69,-74.175), (p_lat,p_lon))
        dropoff_dist_ewr = cityblock((40.69,-74.175), (d_lat,d_lon))
        pickup_dist_lgr = cityblock((40.77,-73.87), (p_lat,p_lon))
        dropoff_dist_lgr = cityblock((40.77,-73.87), (d_lat,d_lon))
        longitude_diff = d_lon - p_lon
        latitude_diff = d_lat - p_lat
        manhattan_distance = cityblock((p_lat,p_lon), (d_lat,d_lon))

        # Adaptive weighting for temp/weather
        actual_distance = get_distance((p_lat, p_lon), (d_lat, d_lon))
        pickup_weight = 0.5 if actual_distance == 0 else min(1, 5/actual_distance)
        avg_temp = pickup_temp * pickup_weight + dropoff_temp * (1 - pickup_weight)
        weather_adj = determine_weather_factor(pickup_condition) * pickup_weight + determine_weather_factor(dropoff_condition) * (1 - pickup_weight)

        # Exactly match training features
        features = np.array([
            p_lat, p_lon, d_lat, d_lon, passenger_count, time.hour,
            date.day, date.month, date.year,
            dist_to_cent,
            pickup_dist_jfk, dropoff_dist_jfk,
            pickup_dist_ewr, dropoff_dist_ewr,
            pickup_dist_lgr, dropoff_dist_lgr,
            longitude_diff, latitude_diff,
            manhattan_distance,
            avg_temp
        ]).reshape(1, -1)

        # Fare calculation precisely matching training logic
        base_fare = abs(model.predict(features)[0])
        final_fare = base_fare * (1 + 0.1 * (passenger_count - 1)) * weather_adj

        # Clearly display accurate results
        st.success(f"✅ Predicted Fare: ${final_fare:.2f}")
        st.success(f"🗺️ Trip Distance: {actual_distance:.2f} miles")

