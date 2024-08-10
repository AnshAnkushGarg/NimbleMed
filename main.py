import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
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
try:
  model = tf.keras.models.load_model('model_4_Bidirectional.keras')
except Exception as e:
  print(e)



st.image("MicrosoftTeams-image (5).png", width=100)
st.markdown("<h1 style='text-align: center; color: #00c9c1;'>NimbleMed+</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #00c9c1;'>We heal because we feel!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00c9c1;'>Need help identiftying your disease? We got you covered! We use NLP and Deep Learning Recurrent Neural Networks to classify your symptoms into diseases!</p>", unsafe_allow_html=True)

st.markdown("<h3 style='color: #00c9c1;'>Symptom to Disease </h3>", unsafe_allow_html=True)

with st.form("symptoms"):
  word = "are"
  st.markdown("<p style='color: #00c9c1;'>For more precise results, please be more specific with your description of the symptomns and write sensible sentences.</p>", unsafe_allow_html=True)

  st.markdown("<h4 style='color: #00c9c1;'>Enter your symptoms: </h4", unsafe_allow_html=True)
  symptom = st.text_input("")
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
      
      index = list(medicines_diseases.keys()).index(key)+1
      
      output = (f"{index}. You could be suffering from disease: {disease} with a probability of {prob}% and the medicine(s) you should take {word} {drugs}.")
      st.markdown("<p style='color: #00f9f1;'>" + output + "</p>", unsafe_allow_html=True)

    st.caption(":blue[Note: source of drugs is from openFDA]")

st.markdown("""
    <style>
    .stForm {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h3 style='color: #00c9c1;'>Nearest Pharmacy Location <br> </h3>", unsafe_allow_html=True)

geo = ('Click the button below so that we can pinpoint your location (click "allow" if any popup appears) and then wait until the nearest pharmacy is displayed <br> <br>')

  
st.markdown("<p style='color: #00f9f1;'>" + geo + "</p>", unsafe_allow_html=True)

location = streamlit_geolocation()
lat = location["latitude"]
lon = location["longitude"]

try:
  pharmacy, loc = get_nearest_pharmacy(lat=lat, lon=lon)

  output = f"Nearest pharmacy near you is: {pharmacy}"

  st.markdown("<p style='color: #00f9f1;'>" + output + "</p>", unsafe_allow_html=True)
except Exception as e:
  print(e)
