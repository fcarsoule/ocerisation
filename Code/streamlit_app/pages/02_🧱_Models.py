#Source : https://github.com/victoryhb/streamlit-option-menu/blob/master/README.md

from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, "../")
import ctcmodel

@st.experimental_singleton
def load_model_ctc(image_width, image_height, vocab_size):

    model, prediction_model = ctcmodel.create_model(image_width, image_height, vocab_size,
                                                    weights_path=None, train=True)
    return model, prediction_model


selected2 = option_menu(None, ["Modelisation", 
                               "Reconnaissance du nombre de caractères", 
                               "Reconnaissance des mots manuscrits"], 
    
    icons=['book','file-bar-graph','file-bar-graph'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2



if selected2=='Modelisation':
   st.title('Présentation du travail de modélisation')
   st.markdown('Dans notre travail de modélisation nous avons abordé deux problématiques principales. Premièrement nous avons créé des modèles visant à détecter le nombre de caractères composant un mot. Deuxièmement, nous avons mis en place des modèles pour la reconnaissance des mots manuscrits.')
   st.markdown('Pour affronter ces deux problématiques nous avons mobilisé et comparé entre eux des modèles de Machine Learning et de Deep Learning. En particulier nous avons mobilisé les les modèles suivants :')
   st.markdown("Dans cette première phase nous avons utilisé comme benchmark un modèle de Random Forest. Par la suite nous avons comparé les performances.")
   st.write(
        """    
                            1. Random Forest.
                            2. Réseau de neurones.
                            3. Réseau de neurones convolutif.
                            4. Modèle LeNet.
        """
    )           
   st.markdown('Compte tenu des limites de ces approches, nous avons mis en place un modèle de Deep Learning plus complexe pour la reconnaissance de mots manuscrits. Ce modèle intègre des couches de convolution, des couches récurrentes et une fonction de perte CTC-Loss. Ce modèle surpasse largement les modèles précédemment employés.')
   st.markdown('Nous allons décrire dans plus de détails toutes les étapes de modélisation ainsi que les performances des différents modèles que nous avons mis en place.')

        
############## Prediction n Characters #################

if selected2=='Reconnaissance du nombre de caractères':
   st.title('Modèles pour la détection du nombre de caractères dans les mots manuscrit')
   st.markdown("Nous avons entamé notre travail avec la mise en place de modèles permettant de reconnaître le nombre de caractères qui composent un mot manuscrit. Nous avons abordé cette problématique comme une problématique de régression.")
   st.markdown("Le modèle de Random Forest a été entraîné avec 90 arbres ayant une profondeur maximale de 5. Les modèles de Deep Learning ont été entraînés sur 50 époques. La métrique que nous avons utilisée pour mesurer les performances de nos modèles et l’ Erreur absolue moyenne")
   st.markdown("Le tableau suivant montre les performances de nos quatre modèles. ")
   
   table_mse = {"Modèles" : ['Random Forest', 'Réseau Dense', 'Réseau de neurones convolutif', 'LeNet'],"Mean absolute error" : [1.3679, 0.8747, 0.6640, 0.4603]}
   table_mse = pd.DataFrame(table_mse)
   st.table(table_mse)
   
   st.markdown("Nous constatons que les modèles de Deep Learning performent beaucoup mieux que le Random Forest.")


############## Prediction Mots #################


if selected2=='Reconnaissance des mots manuscrits':
  model_simpletb,model_complexetb=st.tabs(["Modèles simples", "Modèle avancé"])

  with model_simpletb :

    st.title("Reconnaissance des mots manuscrits")
    st.markdown('Après avoir construit des modèles capables de reconnaître le nombre de caractères des mots manuscrits, nous avons abordé la problématique de la reconnaissance des mots manuscrits.')
    st.markdown("Pour ce faire nous avons limité notre échantillon uniquement aux dix mots le plus fréquents ayant entre 3 et 4 caractères.")
    st.markdown("Le Random Forest a été compilé sans restrictions concernant la profondeur des arbres. Les modèles de Deep Learning ont été construits et compilés comme pour la problématique précédente.")
    st.markdown("La métrique utilisée pour examiner la performance de nos modèles est l’accuracy. Le tableau suivant montre les performances de nos modèles.")
    table_acc = {"Modèles" : ['Random Forest', 'Réseau Dense', 'Réseau de neurones convolutif', 'LeNet'], "Accuracy" : [0.86, 0.86, 0.93, 0.96]}
    
    table_acc = pd.DataFrame(table_acc)
    
    st.table(table_acc)
    
    st.markdown("Nous constatons que les modèles de Deep Learning ont des performances décidément meilleures que le Random Forest. En particulier le modèle LeNet est le modèle ayant les meilleures performances.")
    st.markdown("Le figure suivante montre une Matrice de Confusion pour le modèle LeNet.")
    
    image = Image.open('./pages/confusion_matrix.png')
    st.image(image, caption='LeNet Confusion Matrix')

  with model_complexetb :
    st.markdown("Les performances obtenues avec les modèles précédents sont bonnes. Cependant elles sont limitées à un sous-échantillon limité de notre base de données. En élargissant l'échantillon, les performances de nos modèles se dégradent rapidement.") 
    st.markdown("Afin de reconnaître les mots manuscrits à l’échelle de la totalité de notre base de données nous avons mis en place un modèle plus complexe capable de saisir les liens de séquentialité entre les caractères composant.")
    st.markdown('Ce modèle intègre des couches de convolution permettant d’extraire une séquence de features, des couches récurrentes permettant de saisir les relations de séquentialité et une couche CTC (Connectionist temporal classification) permettant de calculer une fonction de perte adaptée et de décoder l’output du modèle.')
    
    image = Image.open('./pages/ctc_loss.png')
    st.image(image, caption='Exemple de matrice produite par le réseau de neurones et du meilleur chemin pour calculé par la fonction de perte CTC')
    model,_ = load_model_ctc(image_width=128, image_height=32, vocab_size=79)
    with st.expander('View model summary'):
        st.subheader("Model summary")
        model.summary(print_fn=lambda x: st.text(x))
    st.markdown('Avant de composer le modèle il est nécessaire transformer les images de notre base de données : en effet, de manière générale, les modèles OCR opèrent sur des images rectangulaires : nous devons donc transformer les images en préservant le rapport entre les différentes features de l’image et sans affecter le contenu des images. La Figure suivante montre un exemple de transformation de deux mots.')
    
    image = Image.open('./pages/transformation.png')
    st.image(image, caption='Exemple de transformation de deux images')
    
    st.markdown("La métrique utilisée pour évaluer les performances du modèle est l’Edit distance. l’Edit Distance est une mesure de similarité qui quantifie la différence existante entre deux strings en mesurant le nombre minimal d'opérations nécessaires pour transformer une string dans l’autre.")
    st.markdown("Notre modèle reconnaît exactement 73.79% des mots de notre base de données. L'Edit distance moyenne est de 0.45 (et l’Edit distance normalisée est de 0.09). La figure suivant montre un extrait de l’output final de notre modèle pour 5 mots.")
    
    image = Image.open('./pages/model_output.png')
    st.image(image, caption='Output du modèle')