import streamlit as st
from PIL import Image # Manipulation d'images
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np 

st.write("""
# Web application : humpback whale identification
""")

# Chargement de l'image
uploaded_file = st.file_uploader("Upload a file",type=['jpg','png','jpeg'])

if uploaded_file != None:

   image = Image.open(uploaded_file)

   # Importation du train.csv
   data = pd.read_csv('train.csv')
      
   # Informations sur l'image   
   st.write(f"**File name** : {uploaded_file.name}")
   st.write(f"**image shape** : {image.size}")
   id_bal = data[data['Image'] == uploaded_file.name]['Id']
   st.write(f"**Id** : {id_bal.values}")
   st.image(image,width=700,height=700)
   
   # Reformatage de l'image
   def load(np_image):
      np_image = np.array(np_image).astype('float32')/255
      np_image = np.expand_dims(np_image, axis=0)
      return np_image

   # Affichage de l'image
   image = load(image)
   
   st.write("*Inference not available*")
