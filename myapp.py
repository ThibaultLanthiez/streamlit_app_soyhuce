import streamlit as st
from pathlib import Path # Acces aux répertoires
import random # Piocher au hasard des images
import os # Acces aux fonctions systèmes 
from PIL import Image # Manipulation d'images
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from PIL import Image
import numpy as np
#from skimage import transform

st.write("""
# My web application : whales tail
""")

uploaded_file = st.file_uploader("Upload Files",type=['jpg','png','jpeg'])
img = utils.load_image(image_file)
st.image(img,width=250,height=250)


data = pd.read_csv('train.csv')

#imgdir = Path('train/')
#image_name = random.choice(os.listdir(imgdir))
#image = Image.open('train/' + image_name)
    
st.write(f"Nom du fichier : {uploaded_file.name}")
st.write(f"Taille : {image.size}")
id_bal = data[data['Image'] == uploaded_file.name]['Id']
st.write(id_bal)
st.image(image)

#model = load_model('model.hdf5')

#def load(filename):
#   np_image = Image.open(filename)
#   np_image = np.array(np_image).astype('float32')/255
#   np_image = transform.resize(np_image, (224, 224, 3))
#   np_image = np.expand_dims(np_image, axis=0)
#   return np_image

#image = load('train/'+image_name)
#st.write(model.predict(image))

#pred = model.predict_classes(image)

#class_indices = np.load('dict_encodage.npy',
#                        allow_pickle=True).item()

#st.write(f'Prédiction : {list(class_indices.keys())[pred.tolist()[0]]}')
#st.write(f'Num réalité : {class_indices[id_bal.tolist()[0]]}')

