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

model = pickle.load(open("model1.pkl","rb"))


def get_location_by_address(address):
    search = requests.get("https://addresstogps.com/?address="+address)
    soup = BeautifulSoup(search.text,"html.parser")
    lat = soup.findAll(attrs = {"class":"form-control","id":"lat"})
    lon = soup.findAll(attrs = {"class":"form-control","id":"lon"})
    lat1,lon1 =  lat[0]["value"],lon[0]["value"]
    return lat1,lon1

def weather(city):
    city=city.replace(" ","+")
    res = requests.get(f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8',headers=headers)
    soup = BeautifulSoup(res.text,'html.parser')   
    location = soup.select('#wob_loc')[0].getText().strip()  
    time = soup.select('#wob_dts')[0].getText().strip()       
    weather = soup.select('#wob_tm')[0].getText().strip()
    humidity = (soup.select("#wob_hm")[0].getText().strip())
    humidity=float(humidity.replace('%',""))
    temp = round((float(weather))*1.8+ 32,2)
    st.write(location,time)
    #st.write(time)
    st.write("Temperature : ",str(temp)+"°F") 
    return temp



# Title and Logo 
st.image("uber.jpg")
st.title("Prashant Shukla")


# Live Wether 
city = st.text_input("Enter Your City Name :-","New York")
city=city+" weather"
temperature = weather(city)


# Datetime 
date = st.date_input("Date of Pickup")
st.info(date)


# Passenger Count
passenger_count = st.selectbox("Passenger",np.arange(1,7))
st.info(passenger_count)

# hour = st.selectbox("Enter Hours",np.arange(0,24))
# m = st.selectbox("Enter Minutes",np.arange(0,60,5))
time = st.time_input("Enter Pickup Time ",datetime.time(0,00))
st.info(time)

# logitude and latitude for pickup
street = st.text_input("Pickup Location ")
#country = st.text_input("Country:")

lat,lon= get_location_by_address(street)

st.write(lat,lon)
# st.write(float(lat),float(lon))

p_lat = st.number_input("Enter Pickup Latitude ",step=1e-6,format="%.3f")
p_lon = st.number_input("Enter Pickup Longitude",step=1e-6,format="%.3f")


# logitude and latitude for dropoff
street1 = st.text_input("Dropoff Location ")
#country1 = st.text_input("Country:")

lat1,lon1= get_location_by_address(street1)

st.write(lat1,lon1)

d_lat = st.number_input("Enter Dropoff Latitude ",step=1e-6,format="%.3f")
d_lon = st.number_input("Enter Dropoff Longitude",step=1e-6,format="%.3f")


map_data1 = pd.DataFrame({'lat': [p_lat,d_lat], 'lon': [p_lon,d_lon]})
st.map(map_data1)



if st.button("Predict Fare"):

    day,month,year = date.day,date.month,date.year
    dist_to_cent = cityblock((40.7141667, -74.0063889),(p_lat,p_lon))
    pick_dist_to_jfk = cityblock((40.6441666667, -73.7822222222),(p_lat,p_lon))
    drop_dist_to_jfk = cityblock((40.6441666667, -73.7822222222),(d_lat,d_lon))
    pick_dist_to_ewr = cityblock((40.69, -74.175),(p_lat,p_lon))
    drop_dist_to_ewr = cityblock((40.69, -74.175),(d_lat,d_lon))
    pick_dist_to_lgr = cityblock((40.77, -73.87),(p_lat,p_lon))
    drop_dist_to_lgr = cityblock((40.77, -73.87),(d_lat,d_lon))
    long_diff = d_lon - p_lon
    lat_diff =  d_lat - p_lat
    manhattan_dist = cityblock((p_lat,p_lon),(d_lat,d_lon))
    
    
    prediction = np.array([p_lat,p_lon,d_lat,d_lon,passenger_count,time.hour,day,month,year,
                      dist_to_cent,pick_dist_to_jfk,drop_dist_to_jfk,
                     pick_dist_to_ewr,drop_dist_to_ewr,pick_dist_to_lgr,drop_dist_to_lgr,
                       long_diff,lat_diff,manhattan_dist,temperature])
      
    result = model.predict(prediction)
    st.write("The Predicted Fare is :  $",abs(result))


# =========================================================================
# Side bar
data = {"Name" : ["Dixit Dutt Bohra","Lalit Bhaskar Mahale","Manchikatla Raman Kumar","Prashant Shukla","Vipin Kumar Tripathi"],
"PRN ": ["220340128010 ","220340128021","220340128023","220340128036","220340128054"],
"Email ":["dixitduttbohra@gmail.com ","lalitmahale121@gmail.com","ramanmenche@gmail.com","prashantjack.shukla@gmail.com","tripathivipin078@gmail.com"]}
df = pd.DataFrame(data,index = np.arange(1,6))

sidebar = st.sidebar.selectbox("Uber Ride Price Prediction",["","Developer","Guide","Dataset"])

if sidebar == "Developer":
    st.image("developer.jpg")
    st.table(df)

elif sidebar == "Guide":
    st.header("Guide")
    st.image("promod_sir.jpg")
    st.header("""Pramod Kumar Sharma""")
    st.write("""Chief Executive Officer pra-sami \n
             Email : info@prasami.com""")
