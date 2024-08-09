import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import geocoder
from streamlit_geolocation import streamlit_geolocation

def get_nearest_pharmacy(lat, lon):
  key = "f81255b6de0341ac98857d2c8d83f833"

  r = requests.get(f"https://api.geoapify.com/v1/geocode/reverse?lat={lat}&lon={lon}&format=json&apiKey={key}")

  data = r.json()

  location = data["results"][0]["formatted"]
  id = data["results"][0]["place_id"]


  pharmacy_r = requests.get(f"https://api.geoapify.com/v2/places?categories=healthcare.pharmacy&filter=place:{id}&limit=20&apiKey={key}")


  data = pharmacy_r.json()
  print(data)
  properties = data["features"][0]["properties"]
  pharmacy = data["features"][0]["properties"]["formatted"]
  
  return pharmacy, location
  

class_names = ['Psoriasis',
 'Varicose Veins',
 'Typhoid',
 'Chicken pox',
 'Impetigo',
 'Dengue',
 'Fungal infection',
 'COVID-19',
 'Pneumonia',
 'Dimorphic Hemorrhoids',
 'Arthritis',
 'Acne',
 'Bronchial Asthma',
 'Hypertension',
 'Migraine',
 'Cervical spondylosis',
 'Jaundice',
 'Malaria',
 'urinary tract infection',
 'allergy',
 'gastroesophageal reflux disease',
 'drug reaction',
 'peptic ulcer disease',
 'diabetes']

model = tf.keras.models.load_model('model_4_Bidirectional.keras')



st.image("MicrosoftTeams-image (5).png", width=75)
st.markdown("<h1 style='text-align: center; color: #00c9c1;'>NimbleMed+</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #00c9c1;'>We heal because we feel!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00c9c1;'>Need help identiftying your disease? We got you covered! We use NLP and Deep Learning Recurrent Neural Networks to classify your symptoms into diseases!</p>", unsafe_allow_html=True)


with st.form("symptoms"):
  word = "are"
  symptom = st.text_input(":blue[Enter your symptoms]")
  submitted = st.form_submit_button("Submit", type="primary")
  if submitted:
    pred = model.predict(pd.Series([symptom]))
    high_prob_indices = np.where(pred > 0.95)[1]
    high_prob_classes = [class_names[index] for index in high_prob_indices]

    medicines_diseases = {}
    for i in high_prob_classes:
      og_i = i
      i = i.lower()
    
      url = f"https://api.fda.gov/drug/event.json?search=patient.reaction.reactionmeddrapt:{i}"
      response = requests.get(url)
      data = response.json()
      drugs = [i["medicinalproduct"] for i in data["results"][0]["patient"]["drug"]]
      medicines_diseases[i] = {"drugs": drugs, "prob": pred[0][high_prob_indices[high_prob_classes.index(og_i)]]}

    for key in medicines_diseases:
      
      disease = key
      drugs_list = list(set(medicines_diseases[key]["drugs"]))
      prob = round(medicines_diseases[key]['prob']*100, 4)
      drugs = ", ".join(drugs_list).lower()

      if len(drugs_list) == 1:
        word = "is"
      
        
      
      output = (f"You could be suffering from disease {disease} with a probability of {prob}% and the medicine(s) you should take {word} {drugs}.")
      st.markdown("<p style='color: #00f9f1;'>" + output + "</p>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stForm {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)




with st.form("latlong"):
  geo = ('Click the crosshair button so that we can pinpoint your location (click "allow" if any popup appears) and then click "submit" to continue. You can also scroll further and customize the location where you want to find a pharmacy by setting a custom latitude or longitude')

  
  st.markdown("<p style='text-align: center; color: #00f9f1;'>" + geo + "</p>", unsafe_allow_html=True)

  location = streamlit_geolocation()
  lat = location["latitude"]
  lon = location["longitude"]

  submit = st.form_submit_button("Submit", type="primary")

  if submit:
    try:
      pharmacy, loc = get_nearest_pharmacy(lat=lat, lon=lon)

      output = f"Nearest pharmacy near you is: {pharmacy} <br> Customize Latitude and Longitude here:"

      st.markdown("<p style='color: #00f9f1;'>" + output + "</p>", unsafe_allow_html=True)
    except Exception as e:
      print(e)

with st.form("latloncustom"):
    lat = st.number_input(":blue[Latitude]", value=lat, step=0.1, format="%.4f")
    lon = st.number_input(":blue[Longitude]", value=lon, step=0.1, format="%.4f")
    submitted = st.form_submit_button("Submit", type="primary")
  
    if submitted:
      pharmacy, loc = get_nearest_pharmacy(lat=lat, lon=lon)
      
      output = f"Nearest pharmacy near {loc} is {pharmacy}"
      st.markdown("<p style='color: #00f9f1;'>" + output + "</p>", unsafe_allow_html=True)
  