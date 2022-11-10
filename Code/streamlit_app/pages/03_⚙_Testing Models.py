#Source : https://github.com/victoryhb/streamlit-option-menu/blob/master/README.md

from streamlit_option_menu import option_menu
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from os import listdir, getcwd
from os.path import isfile, join, isdir
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

import sys
sys.path.insert(0, "../")

from importation import Importation
from preprocessor import Preprocess
import ctcmodel
import ctcprediction
import top10model
import top10prediction
import wsmodel
import wsprediction

data = "words"

if isdir('../database'):
    imp = Importation('../database')
else : imp = Importation('database')
if isdir('../models'):
    weights_folder ="../models/"
else : weights_folder ="./models/"

titre = st.title("Testing Models")
genre = None
selectModel = None
model = None 

# Chargement des données
@st.cache
def load_data(data):
    assert data=="words" or data=="lines"
    image_width, image_height = 0, 0
    if data=="words":
        df = imp.get_words().sample(frac=1, random_state=1).reset_index()
        image_width = 128
        image_height = 32
    else :
        df = imp.get_lines().sample(frac=1, random_state=1).reset_index()
        image_width = 400
        image_height = 32
    max_len, longest_word, characters = imp.get_characters(df.label)
    vocab_size = len(characters)
    return df.sample(frac=0.1), image_width, image_height ,max_len, longest_word, characters, vocab_size

@st.cache
def load_data_top10(data):
    df = imp.get_words().sample(frac=1, random_state=1).reset_index()
    words_list = df[(df.word_size==3)|(df.word_size==4)].label.value_counts().head(10)
    df= df[df.label.isin(words_list.index)]
    image_width, image_height = 0, 0
    if data=="words":
        image_width = 128
        image_height = 32
    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df.label)

    return df, image_width, image_height, encoder

@st.cache
def load_images_test(data):
    if data=="words":
        rep = "database/images_test/words"
    fichiers = sorted([(join(rep, f), f) for f in listdir(rep) if isfile(join(rep, f))])
    return fichiers

@st.experimental_singleton
def load_model_ctc(data, image_width, image_height, vocab_size):
    if data=="words":
        weights_path = weights_folder +"model_ctc_words.h1"
    elif data=="lines":
        weights_path = weights_folder +"model_ctc_lines.h1"
    model, prediction_model = ctcmodel.create_model(image_width, image_height, vocab_size, weights_path)
    return model, prediction_model

@st.experimental_singleton
def load_model_top10(image_width, image_height, model_type):
    num_classes = 10
    if model_type=="Dense":
        weights_path = weights_folder +"top10_dense.h1"
        model = top10model.create_model_dense(image_width, image_height, num_classes)
        model.load_weights(weights_path)
    if model_type=="CNN":
        weights_path = weights_folder +"top10_cnn.h1"
        model = top10model.create_model_cnn(image_width, image_height, num_classes)
        model.load_weights(weights_path)
    if model_type=="LeNet":
        weights_path = weights_folder +"top10_lenet.h1"
        model = top10model.create_model_lenet(image_width, image_height, num_classes)
        model.load_weights(weights_path)
    if model_type=="RF":
        model_path = weights_folder +"top10_rf.sav" 
        model = pickle.load(open(model_path, 'rb'))
       
    return model

@st.experimental_singleton
def load_model_ws(image_width, image_height, model_type):

    if model_type=="Dense":
        weights_path = weights_folder +"word_size_dense.h1"
        model = wsmodel.create_model_dense(image_width, image_height)
        model.load_weights(weights_path)
    if model_type=="CNN":
        weights_path = weights_folder +"word_size_cnn.h1"
        model = wsmodel.create_model_cnn(image_width, image_height)
        model.load_weights(weights_path)
    if model_type=="LeNet":
        weights_path = weights_folder +"word_size_lenet.h1"
        model = wsmodel.create_model_lenet(image_width, image_height)
        model.load_weights(weights_path)
    return model

def predictws(df_predict, models, prepro):
    labels = []
    predictions = []
    images = []
    # for i in np.random.choice(np.arange(0, len(df_predict)), size=3):
    for i in range(len(df_predict)):
        label = df_predict.iloc[i].label
        labels.append(label)
        word_size = df_predict.iloc[i].word_size
        img_path = df_predict.iloc[i].file
        ds = wsprediction.get_ds(img_path, word_size, prepro)
        preds = []
        for model in models:
            prediction= wsprediction.get_prediction(ds, model[0])
            prediction = np.round(prediction,2)[0]
            preds.append((prediction, model[1]))
        predictions.append(preds)
        images.append(cv2.imread(img_path))     
    return images, labels, predictions

def predicttop10(df_predict, models, prepro, encoder):
    labels = []
    predictions = []
    probabilities = []
    images = []
    # for i in np.random.choice(np.arange(0, len(df_predict)), size=3):
    for i in range(len(df_predict)):
        label_encoded = df_predict.iloc[i].label_encoded
        label = df_predict.iloc[i].label
        labels.append(label)
        img_path = df_predict.iloc[i].file
        ds = top10prediction.get_ds(img_path, label_encoded, prepro)
        preds = []
        probas = []
        for model in models:
            prediction, proba, inputs = top10prediction.get_prediction(ds, model[0])
            preds.append((prediction, model[1]))
            probas.append(proba)
            
        predictions.append(preds)
        probabilities.append(probas)
        images.append(cv2.imread(img_path))     
    return images, labels, predictions, probabilities

def predictctc(df_predict, model, prepro):
    labels = []
    predictions = []
    distances = []
    images = []
    for i in range(len(df_predict)):
        label = df_predict.iloc[i].label
        labels.append(label)
        img_path = df_predict.iloc[i].file
        x = np.array([img_path])
        y = np.array([label])
        ds = prepro.prepare_dataset(x, y)
        prediction, distance, preds = ctcprediction.get_prediction(ds, model, prepro)         
        predictions.append(prediction)
        distances.append(distance)
        images.append(cv2.imread(img_path))     
    return images, labels, predictions, distances

def predictctc_bs(df_predict, model, prepro, top_paths=3, beam_width=1000):
    labels = []
    predictions = []
    probas = []
    #distances = []
    images = []
    for i in range(len(df_predict)):
        label = df_predict.iloc[i].label
        labels.append(label)
        img_path = df_predict.iloc[i].file
        x = np.array([img_path])
        y = np.array([label])
        ds = prepro.prepare_dataset(x, y)
        prediction, proba = ctcprediction.get_prediction_bs(ds, model, prepro,
                                                            top_paths=top_paths,
                                                            beam_width=beam_width)
        predictions.append(prediction)
        probas.append(proba)
    return images, labels, predictions, probas

def predictctc_np(image, model, prepro):

    label = ""

    x = [image]
    y = np.array([label])
    ds = prepro.prepare_dataset_np(x, y)
    prediction, distance, preds = ctcprediction.get_prediction(ds, model, prepro)         

        
    return  prediction

def predictctc_np_bs(image, model, prepro, top_paths=10, beam_width=1000):

    label = ""

    x = [image]
    y = np.array([label])
    ds = prepro.prepare_dataset_np(x, y)
    predictions, logprobas = ctcprediction.get_prediction_bs(ds, model, prepro,
                                                        top_paths=top_paths,
                                                        beam_width=beam_width)         

        
    return  predictions, logprobas

_wscl = "Reconnaissance du nombre de caractères - classification"
_wsreg = "Reconnaissance du nombre de caractères"
_top10 = "Reconnaissance de mots manuscrits - Top 10"
_ctc = "Reconnaissance de mots manuscrits"

_data_words = "Words"
_data_lines = "Lignes"



with st.sidebar:
   # add_SelectModel = st.selectbox("Select a Model",("Model1","Model2","Model3"))
    genre = st.radio("Modélisation",
                     (_wsreg ,
                      _top10,
                      _ctc)
                     )
    # if (genre==_ctc):
    #     data_type = st.radio("Mots ou Lignes",
    #                          (_data_words, _data_lines))
    #     data = "words" if data_type==_data_words else "lines"


# Création menu
if (genre==_wscl or genre==_wsreg or genre==_top10):
    selected2 = option_menu(None, ['Random Image'], 
        icons=['file-earmark-text'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
elif genre==_ctc:
    selected2 = option_menu(None, ['Random Image', 'Write Text', 'Load Image'], 
        icons=['file-earmark-text', 'image-fill','inbox'], 
        menu_icon="cast", default_index=0, orientation="horizontal")


# Chargement des données

if (genre==_wscl or genre==_wsreg or genre==_ctc):   
    #  data_load_state = st.text('Chargement des données...')
    df, image_width, image_height ,max_len, longest_word, characters, vocab_size = load_data(data)

    #data_load_state.text('Chargement des données...fait!')
elif genre==_top10:
    df, image_width, image_height, encoder = load_data_top10("words")
    with st.expander("View Top 10 words", expanded=False):
        st.dataframe(df['label'].value_counts())
        st.markdown(str(set(df['label'])).replace('{','').replace('}', '').replace(',',', ').replace('\'',''))
img_size = (image_width, image_height) 
   

############## Random Image ################
if selected2=='Random Image':
    st.info('Prédiction sur des images prises au hasard dans la base')

    #number = st.number_input('Insert a number', max_value=10, min_value=1, value=3)
    number = st.slider("Sélectionner le nombre d'images à afficher", 1, 10, 3)
    

    if genre==_wsreg:
        size = st.slider("Sélectionner la taille des images", 1, 14, 5)
        df_predict = df[df['word_size']==size].sample(number)
        model_dense = load_model_ws(image_width, image_height, "Dense")
        model_cnn = load_model_ws(image_width, image_height, "CNN")
        model_lenet = load_model_ws(image_width, image_height, "LeNet")
    
    
        batch_size=5
        prepro = Preprocess(img_size, batch_size=batch_size, gaussianBlur=True)
        models = [(model_dense, "Réseau de neurones artificiels"),
                  (model_cnn, "Réseau de neurones convolutifs"),
                  (model_lenet, "Réseau LeNet")]
        images, labels, predictions = predictws(df_predict, models, prepro)
        # st.image(images, caption=predictions)
        st.write()
        tableaux = [pd.DataFrame(columns=["modèle", "taille réelle", "Prédiction", "Erreur"])]*len(images)
        for i in range(len(images)):
           # with st.container():
            caption = f"{labels[i]}"
            
            #col1, col2, col3, col4 = st.columns([1,2,1,1])
            col1, col2 = st.columns([1,2])
            #col2.subheader("Prédictions")
            with col1:
                st.subheader(f"image {i+1}")
                st.write(caption)
                st.image(images[i], caption=caption)


            with col2:
                for j, prediction, in enumerate(predictions[i]):
    
                    tableaux[i].loc[j] = [f"{prediction[1]}",
                                          f"{len(labels[i])}",
                                           f" {prediction[0]}",
                                           f"{np.round(np.abs(prediction[0] - len(labels[i])),2)}"]

                tableaux[i]    
            st.markdown("<hr>", unsafe_allow_html=True)

    elif genre==_wscl:
        st.info("en construction....")
    elif  genre==_top10:
        
        df_predict = df.sample(number)
        model_dense = load_model_top10(image_width, image_height,"Dense")
        model_cnn = load_model_top10(image_width, image_height,"CNN")
        model_lenet = load_model_top10(image_width, image_height,"LeNet")
        #model_rf = load_model_top10(image_width, image_height,"RF")
        
    
        batch_size=5
        prepro = Preprocess(img_size, batch_size=batch_size, gaussianBlur=True)
        models = [#(model_rf, "Random Forest"),
                  (model_dense, "Réseau de neurones artificiels"),
                  (model_cnn, "Réseau de neurones convolutifs"),
                  (model_lenet, "Réseau LeNet"),
                 ]
        images, labels, predictions, probabilities = predicttop10(df_predict, models, prepro, encoder)
        # st.image(images, caption=predictions)
        tableaux = [pd.DataFrame(columns=["modèle", "Prédiction", "Probabilité"])]*len(images)

        for i in range(len(images)):

            caption = f"{labels[i]}"
            
            #col1, col2, col3, col4 = st.columns([1,2,1,1])
            col1, col2 = st.columns([1,2])
            #col2.subheader("Prédictions")
            with col1:
                st.subheader(f"image {i+1}")
                st.write(caption)
                st.image(images[i], caption=caption)

            with col2:
                for j, (prediction, proba) in enumerate(zip(predictions[i], probabilities[i])):
    
                    tableaux[i].loc[j] = [f"{prediction[1]}",
                                           f" {encoder.inverse_transform(prediction[0])[0]}",
                                           f"{np.round(proba[0],2)}"]

                tableaux[i]    
            st.markdown("<hr>", unsafe_allow_html=True)
            
            
            
    elif genre==_ctc:
        _, model_ctc = load_model_ctc(data, image_width, image_height, vocab_size)
        if data=="words":
            df_predict = df[df['word_size']>3].sample(number)
        else : df_predict = df.sample(number)
            
    
        batch_size=1
        prepro = Preprocess(img_size, batch_size=batch_size, gaussianBlur=True,
                            max_len=max_len, characters=characters)
        
        images, labels, predictions, distances =predictctc(df_predict, model_ctc, prepro)
        _ ,_ ,predictions_bs, logprobas = predictctc_bs(df_predict, model_ctc, prepro, top_paths=3)
        
        tableaux = [pd.DataFrame(columns=["Prédiction", "Edit distance", "Edit distance normalisée"])]*len(images)
        tableaux_bs = [pd.DataFrame(columns=["Prédiction", "Probabilité"])]*len(images)
        for i in range(len(images)):
            caption = f"{labels[i]}"          
            col1, col2 = st.columns([1,2])
            with col1:
                st.subheader(f"image {i+1}")
                st.image(images[i], caption="")
                st.markdown("<h3 style='color:darkblue;'>"+caption+"</h3>", unsafe_allow_html=True)
                
            with col2:
                for j, (prediction, distance) in enumerate(zip(predictions[i], distances[i])):
                    tableaux[i].loc[j] = [f" {prediction}",
                                           f"{int(distance)}",
                                           f"{np.round(distance / len(labels[i]),2)}"
                                           ]
                st.table(tableaux[i])

                probs = tf.math.exp(logprobas[i][0]).numpy()
                probs = probs[0]
                for l, pred in enumerate(predictions_bs[i][0]):
                    tableaux_bs[i].loc[l] = [ctcprediction.decode_to_text(pred, prepro)[0],
                                             probs[l]]

                st.table(tableaux_bs[i].style.format({"Probabilité": "{:.3f}"}))             
            st.markdown("<hr>", unsafe_allow_html=True)


############## Load Image ################
if selected2=='Load Image':
    # st.markdown('Identifier caractere depuis un texte scanné')
    file_png = st.file_uploader("Upload a PNG image", type=([".png"]))
    
    if genre==_ctc:

        _, model_ctc = load_model_ctc(data, image_width, image_height, vocab_size)
        
        if file_png:
            if file_png is not None:
                image = Image.open(file_png)
                img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(img)
            
            if np.min(img)!=np.max(img):
                batch_size = 1
                prepro = Preprocess(img_size, batch_size=batch_size, gaussianBlur=True, max_len=max_len, characters=characters)   
                
                img = tf.expand_dims(tf.convert_to_tensor(img), axis=2)

                prediction = predictctc_np(img, model_ctc, prepro) 
                prediction_bs, logprobas = predictctc_np_bs(img, model_ctc, prepro) 
                probs = tf.math.exp(logprobas[0]).numpy()
                probs = probs[0]
                tableau_bs = pd.DataFrame(columns=["Prédiction", "Probabilité"])
                for l, pred in enumerate(prediction_bs[0]):
                    tableau_bs.loc[l] = [ctcprediction.decode_to_text(pred, prepro)[0],
                                             probs[l]]
                col1, col2 = st.columns([1,4])
                with col1:
                    st.info("Prédiction :")
                with col2:
                    st.success(prediction[0])
                    st.table(tableau_bs.style.format({"Probabilité": "{:.3f}"}))

    


############## Write Text #################

if selected2=='Write Text':
    st.info('Identifier caractere depuis une saisie manuelle dans un canvas')
    ##Code Source for Hand Writing Canvas https://github.com/ai-14/deep-detect-my-handwriting/blob/master/src/main.py

    # Create a canvas component

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=10,
        stroke_color='black',
        background_color='lightgrey',
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=False,
        height=150,
        drawing_mode="freedraw",
        point_display_radius=0,
        key="canvas",
        display_toolbar=True
    )
        
    if genre==_ctc:
        data = "words"
        _, model_ctc = load_model_ctc(data, image_width, image_height, vocab_size)

        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.min(img)!=np.max(img):
                batch_size = 1
                prepro = Preprocess(img_size, batch_size=batch_size, gaussianBlur=True, max_len=max_len, characters=characters)   
                
                img = tf.expand_dims(tf.convert_to_tensor(img), axis=2)

                prediction = predictctc_np(img, model_ctc, prepro) 
                prediction_bs, logprobas = predictctc_np_bs(img, model_ctc, prepro) 
                probs = tf.math.exp(logprobas[0]).numpy()
                probs = probs[0]
                tableau_bs = pd.DataFrame(columns=["Prédiction", "Probabilité"])
                for l, pred in enumerate(prediction_bs[0]):
                    tableau_bs.loc[l] = [ctcprediction.decode_to_text(pred, prepro)[0],
                                             probs[l]]
                col1, col2 = st.columns([1,4])
                with col1:
                    st.info("Prédiction :")
                with col2:
                    st.success(prediction[0])
                    st.dataframe(tableau_bs.style.format({"Probabilité": "{:.3f}"}))
            else:
                st.write("canvas vide")

if not genre is None :
    titre.title(genre)
    



