#Source : https://github.com/victoryhb/streamlit-option-menu/blob/master/README.md

####################IMPORT PACKAGE##############################
from streamlit_option_menu import option_menu
import streamlit as st
import matplotlib
import squarify
import plotly.express as px
import plotly.graph_objects as go
import os
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from wordcloud import WordCloud
import pandas as pd
import cv2
import numpy as np
#from google.colab.patches import cv2_imshow
from PIL import Image
import random


#######################SET OPTION################
st.set_option('deprecation.showPyplotGlobalUse', False)

###############Import fichier ASCII ###################################
import sys
sys.path.insert(0, "../")

from importation import Importation
if isdir('../database'):
    imp = Importation('../database')
else : imp = Importation('database')
#imp = Importation("database")

@st.cache
def load_words_err():
    words_err= imp.get_words(drop_err_seg=False)
    return words_err

@st.cache
def load_words():
    words= imp.get_words(drop_err_seg=True)
    return words

@st.cache
def load_lines_err():
    lines_err=imp.get_lines(drop_err_seg=False)
    return lines_err

@st.cache
def load_sentences_err():
    sentences_err=imp.get_sentences(drop_err_seg=False)
    return sentences_err

@st.cache
def load_forms():
    forms=imp.get_forms_ini()
    return forms


#############################FIN###################################
st.set_page_config(page_title="Introduction", 
                   page_icon=None, 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

forms = load_forms()
sentences_err = load_sentences_err()
lines_err = load_lines_err()
words_err = load_words_err()
words = load_words()

     
selected2 = option_menu(None, ["Contexte", "Exploration des Données"], 
    icons=['book','bar-chart-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2


############## Introduction #################


if selected2=='Contexte':

    
   # Tabs
    contexttb,DATAtb=st.tabs(["Contexte", "Jeux de Données : IAM Handwriting Database"])
    #add_selectText = st.radio("",("Context", "Dataset : IAM Handwriting Database"))
   
   
    with contexttb:   
   # Header
       st.header ("Introduction")
       st.markdown("Objectif de ce projet OCR est de prédire les mots à partir de mots manuscrits \
       enregistrés au format image (.png)")
       st.markdown("Le jeux de Données utilisé est la  [IAM Handwriting Database] (vocabulaire anglais)."\
                   "(https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)")
       st.markdown(" Une description complète de ce jeux de donnés sera présentée.")
   # Header
       st.header ("Ocerisation : Principes")
       st.markdown("Ocerisation consistes en la détection de mots à partir d'un texte écrit ou scanné \
       ou encore une image.")
       st.markdown("Ocerisation: Cette méthode peut être décrite selon les prncipales étapes suivantes:")
       st.write(
        """    
                            1. Saisie image.
                            1. Pre processing image.
                            2. Détection/échantillonage du texte en utilisant ma méthode des Bounding boxes.
                            3. Traduction des images en textes.
                            4. Texte prédit.
        """
    )            
       
       ##Lecture GIF
       import streamlit as st
       import base64
       
       #https://global.discourse-cdn.com/standard17/uploads/hellohellohello/original/2X/7/73dbaceeef67642517a76f86668511461472b0dd.gif
       
       import os 
       import requests
       url = 'https://global.discourse-cdn.com/standard17/uploads/hellohellohello/original/2X/7/73dbaceeef67642517a76f86668511461472b0dd.gif'
       page = requests.get(url)
       f_ext = os.path.splitext(url)[-1]
       f_name = 'img{}'.format(f_ext)
       with open(f_name, 'wb') as f: 
           f.write(page.content)

       """### Ocerisation : Example"""
       file_ = open(f_name, "rb")
       contents = file_.read()
       data_url = base64.b64encode(contents).decode("utf-8")
       file_.close()

       st.markdown(
           f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
           unsafe_allow_html=True,)
       

   
    with DATAtb:   
    # Header
        st.header ("Jeux de Données: description")
        st.markdown("IAM Handwriting a été publiée pour la première fois en **1999**" \
                    ".Depuis, des ajouts et mises à jours régulières sont apportées \
                    jusqu'à encore aujourd'hui.")
        st.markdown("Le jeu de données présentes des formes non contraintes, "\
                    "qui ont été scannées à une résolution de 300 dpi et enregistrées en tant qu'image PNG" \
                        " avec 256 niveaux de gris")
        col1,col2 = st.columns(2)
        with st.expander("Jeux de Données: description"):
            with col1:
               st.subheader("Détails")
               st.write("**657** auteurs")
               st.write("**1539** pages scannées")
               st.write("**5685** phrases " )
               st.write("**13353** lignes " )                    
               st.write("**115320** mots " ) 
            with col2:
                st.subheader("Database: Contenu")
                st.write("**Fichiers ASCII** *Forms*,*Sentences*,*Lines*, *Words*")
                st.write("**PNG Files** *scans des formes, lignes, phrases et mots*")
        st.header ("ASCII FILES CONTENU")

        def _descAscii(_df):
          _data={'Columns': _df.columns}
          _desc=pd.DataFrame(_data)
          for idx in _desc.index:
            if idx==0 :
                _desc.loc[idx,'Description']="Primary Key"
            elif _desc.loc[idx,'Columns'] =='x':
                    _desc.loc[idx,'Description']="Bounding box : x value"
            elif _desc.loc[idx,'Columns'] =='y':
                    _desc.loc[idx,'Description']="Bounding box : y value"
            elif _desc.loc[idx,'Columns'] =='w':
                    _desc.loc[idx,'Description']="Bounding box : width value"
            elif _desc.loc[idx,'Columns'] =='h':
                    _desc.loc[idx,'Description']="Bounding box : height value"
            elif idx !=0 and _desc.loc[idx,'Columns'].find('_id')>0 :
                    _desc.loc[idx,'Description']="Foreign Key"
            elif _desc.loc[idx,'Columns'] =='result_w_seg':
                    _desc.loc[idx,'Description']="Segmentation quality ('ok' or 'err')"
            elif _desc.loc[idx,'Columns'] =='grammatical tag':
                    _desc.loc[idx,'Description']="Grammatical tag" 
            elif _desc.loc[idx,'Columns'] =='label':
                    _desc.loc[idx,'Description']="Scanned text"
            elif _desc.loc[idx,'Columns'].find('number_of')>=0:
                _desc.loc[idx,'Description']=_desc.loc[idx,'Columns'].replace("_"," ").title()
          _desc=_desc.dropna()
          return st.dataframe(_desc)
        rad_selectASCII = st.radio("Choisir Fichier ASCII ",("Forms","Sentences","Lines","Words" ),horizontal=True)
        col1,col2 = st.columns(2)
        if rad_selectASCII=="Forms":
            with col1:
                st.subheader("Metadonnées")
                _descAscii(forms)
            with col2:
                st.subheader("Database")
                st.dataframe(forms.head())

        if rad_selectASCII=="Sentences":
            with col1:
                st.subheader("Metadonnées")
                _descAscii(sentences_err)
            with col2:
                st.subheader("Database")
                st.dataframe(sentences_err.head())
                
        if rad_selectASCII=="Lines":
            with col1:
                st.subheader("Metadonnées")
                _descAscii(lines_err)            
            with col2:
                st.subheader("Database")
                st.dataframe(lines_err.head())
 
        if rad_selectASCII=="Words":
            with col1:
                st.subheader("Metadonnées")
                _descAscii(words_err)               
            with col2:
                st.subheader("Database")
                st.dataframe(words_err.head())
             
            
        
        

############## Data Exploration #################

if selected2=='Exploration des Données':

    segmentationtb,freqtb, samplingtb=st.tabs(["Segmentation","Frequence","Echantillonnage"])
    
    #####SEGMENTATION##################
    with segmentationtb:
        st.markdown("Segmentation : Extraction texte via outils Bounding boxes\
                    [(Bibliographie)]" "(https://www.researchgate.net/publication/220931722_Automatic_Segmentation_o_the_IAM_Off-Line_Database_orHandwrittenEnglishText)") 
        st.subheader("Taux Qualité Segmentation")
        
        ##########Segmentation Graph##########
        
        ##Word##
        #import fichier ascii
        col1,col2,col3 = st.columns(3)
        
        #fonction pie
        @st.cache(suppress_st_warning=True)
        def _editpie(_file,_item,label):
          _ok,_err=_file[_item].value_counts(normalize=True)[0],_file[_item].value_counts(normalize=True)[1]
        
          labels = ['Ok','Error']
          values = [_ok,_err]
          fig = go.Figure(data=[go.Pie( labels=labels, values=values, name="Segmentation Quality Rate")])
          # Use `hole` to create a donut-like pie chart
          fig.update_traces(hole=.5, hoverinfo="label+percent+name",textinfo='none')
          fig.update_layout(
                    title_text=  label,
                    # Add annotations in the center of the donut pies.
                    annotations=[dict(text=str(round(_ok*100,0)) + '%', x=0.5, y=0.5, font_size=20, showarrow=False)])
          st.plotly_chart(fig,use_container_width=True)
        
        def segerror(df):
          _tmperr= df[df['result_w_seg']=='err'][['writer_id','label','file']]
          _tmpOK=df[df['result_w_seg']=='ok'][['writer_id','label','file']]
          _err=pd.merge(_tmperr,_tmpOK,how='inner', on=['writer_id','label'], suffixes=('err','ok'))
          _err.drop_duplicates(subset=['label'], inplace=True)
            
          _index=np.random.choice(_err.index)
          fileerr=_err[_err.index==_index]['fileerr'].iloc[0]
          fileok=_err[_err.index==_index]['fileok'].iloc[0]
          _label=_err[_err.index==_index]['label'].iloc[0]
          imageERR=cv2.imread(str(fileerr), cv2.IMREAD_GRAYSCALE)
          imageOK=cv2.imread(str(fileok), cv2.IMREAD_GRAYSCALE)
          st.write( "Label = " + _label) 
          fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,30),constrained_layout = True)
          ax[0].set_title("Image Segmentation OK")
          ax[0].imshow(imageOK,cmap='gray')
          ax[1].set_title("Image Segmentation Error")
          ax[1].imshow(imageERR,cmap='gray')
          st.pyplot()
        
        with col1:
            _editpie(words_err,'result_w_seg',"Words")
            if st.button("Voir Exemples Words"):
                segerror(words_err)
                
            
        with col2:            
            _editpie(lines_err,'result_w_seg',"Lines")
            if st.button("Voir Exemples Lines"):
                segerror(lines_err)
        with col3:            
            _editpie(sentences_err,'result_w_seg',"Sentences")
            #if st.button("View Examples Sentences"):
                #segerror(sentences_err)
    with freqtb:
            selectViz=st.radio("",("Mot le plus Fréquent","Auteur le plus fréquent"\
                                          ,"Nombre de lettres/mot")\
                           ,horizontal=True)
            if selectViz=="Mot le plus Fréquent":
                _numTopWrd= st.radio(f"Nombre de mots à présenter \
                (Total Nombre de mots distincts {len(words['label'].unique())})",[5,10,20,30,50,100],horizontal=True,label_visibility="visible")
                _tmp=words_err.loc[words_err['result_w_seg']=="ok"]['label'].value_counts()
                _label=words_err.loc[words_err['result_w_seg']=="ok"]['label']
                #Table des frequences de mots
                _freq={}
                for _ind,_lab in zip(_tmp.to_frame().index,_tmp.to_frame().label):
                  _freq[_ind]=_lab
                
                wc = WordCloud(background_color="white", max_words=_numTopWrd, max_font_size=50, random_state=42).generate_from_frequencies(_freq)

                # Générer et afficher le nuage de mots
                
                plt.figure(figsize= (10,6)) # Initialisation d'une figure
                plt.axis("off")
                plt.imshow(wc)
                st.pyplot()

            if selectViz=="Auteur le plus fréquent":
                _TopAuth=st.radio(f"Premiers Contributeurs \
                (Total Nombre de Contributeurs distincts {len(words['writer_id'].unique())})",[10,30,50,100],\
                horizontal=True,label_visibility="visible")
                _tmp1=words_err['writer_id'].value_counts(normalize=True).to_frame()
                _tmp1['_percent']=round(_tmp1['writer_id']*100,2)
                _tmp=_tmp1.iloc[:_TopAuth]
                x = 0.
                y = 0.
                width = 100.
                height = 100.
                    
                cmap = matplotlib.cm.plasma
                    
                    # color scale on the population
                    # min and max values without Pau
                mini, maxi = _tmp._percent.min(), _tmp._percent.max()
                norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
                colors = [cmap(norm(value)) for value in _tmp._percent]
                colors[1] = "#FBFCFE"
                    
                # labels for squares
                labels = [label + "\n" + "( " +str(_value) +"%)" for label,_value in zip(_tmp.index, _tmp._percent)]
                    
                # make plot
                fig = plt.figure(figsize=(30, 30))
                ax = fig.add_subplot(111, aspect="equal")
                ax = squarify.plot(_tmp._percent, color=colors, label=labels, ax=ax, alpha=.7)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Square area correspond to Author contribution (in %) \n", fontsize=14)
                    
                # color bar
                # create dummy invisible image with a color map
                img = plt.imshow([_tmp._percent], cmap=cmap)
                img.set_visible(False)
                fig.colorbar(img, orientation="vertical", shrink=.96)
                st.pyplot(fig)
                
                
            if selectViz=="Nombre de lettres/mot":                
                @st.cache(suppress_st_warning=True)
                def _pareto(_df,_var,_label):
                    _dfC = _df[_var].astype('str').value_counts().sort_values( ascending=False).to_frame()
                
                    #add column to display cumulative percentage
                    _dfC['cumperc'] = _dfC[_var].cumsum()/_dfC[_var].sum()*100
                
                    #define aesthetics for plot
                    color1 = 'steelblue'
                    color2 = 'red'
                    line_size = 4
                
                    #create basic bar plot
                    fig, ax = plt.subplots(constrained_layout = True)
                    ax.bar(_dfC.index, _dfC[_var], color=color1)
                
                    #add cumulative percentage line to plot
                    ax2 = ax.twinx()
                    ax2.plot(_dfC.index, _dfC['cumperc'], color=color2, marker="D", ms=line_size)
                    ax2.yaxis.set_major_formatter(PercentFormatter())
                
                    #specify axis colors
                    ax.set(xlabel='Number of characters/Words', ylabel='Number of Words')
                    ax2.set(ylabel='Cumulated percentage(%)')
                    ax.tick_params(axis='y', colors=color1)
                    ax2.tick_params(axis='y', colors=color2)
                
                    #display Pareto chart
                    plt.title(_label)
                    st.pyplot()
                    
                col1,col2=st.columns(2)
                with col1:
                    _pareto(words,'word_size','Distribution taille des mots (all)')
                with col2:
                    _df=words.drop_duplicates('label')
                    _pareto(_df,'word_size','Distribution taille des mots (Drop duplicate)')
                    
                    
        
    with samplingtb:
            selectSampling=st.radio("",("Comparaison Mots pour un même Auteur","Distribution des mots")\
                           ,horizontal=True)
            if selectSampling=="Comparaison Mots pour un même Auteur":
                        _selectoccurence=st.slider(label="Selectionner Occurence d'un mot",\
                        min_value=1,value=3,\
                        max_value=214,step=1)
                        _selectlen=st.slider(label="Selectionner Longueur d'un mot",\
                        min_value=1,value=3,\
                        max_value=16,step=1)  
                        col1,col2 = st.columns(2)
                        @st.cache(suppress_st_warning=True)
                        def _compareWords(writerid,_len,_cnt):
                          __dfc=words[words['writer_id']==writerid].groupby(by=['label','writer_id'],group_keys=False).count().rename(columns={'word_id':'count'}).reset_index()
                          __dfc['nbcar']=__dfc[['label','writer_id','count']]['label'].apply(lambda x: len(x))
                          _dfc=pd.merge(__dfc[['label','writer_id','count','nbcar']], \
                          words[['file','label','writer_id']],how='inner',on=['label','writer_id'])

                          _lengthselect =_len
                          _countselect =_cnt
                          _dfsubselect=_dfc[(_dfc['count']<=_countselect) & (_dfc['nbcar']==_lengthselect)]
                          

                          if not(_dfsubselect.empty):
                             _tmplabel=np.random.choice(_dfsubselect['label'])

                             _FileRef=_dfc[_dfc['label']==_tmplabel]['file'].iloc[0]
                             #XRef=[]
                             imageR=cv2.imread(str(_FileRef), cv2.IMREAD_GRAYSCALE)
                             #XRef.append(np.array(imageR))
                             fig, ax = plt.subplots(nrows=len(_dfc[_dfc['label']==_tmplabel]['file']) \
                             , ncols=2, figsize=(10,8*len(_dfc[_dfc['label']==_tmplabel]['file'])),constrained_layout = True)
                             j=0
                             st.write("Writer : 000" + "/ Text : " +  _tmplabel )
                             for i, axi in enumerate(ax.flat):
                                #X=[]

                                if i%2==0:
                                    axi.set_title("Reference image")                                    
                                #st.image(_imageRef)
                                    axi.imshow(imageR,cmap='gray')
                                else:
                                    file = _dfc[_dfc['label']==_tmplabel]['file'].iloc[j]
                                    if file != _FileRef :
                                        axi.set_title("Compared image")
                                #st.image(file)
                                        image=cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                                #X.append(np.array(image))
                                        axi.imshow(image,cmap='gray')
                                    j+=1
                             st.pyplot()
                          else : st.write("Pas d'occurence")
                          return _tmplabel
                         
                        with col1:

                            _randomAuthor=_compareWords("000",_selectlen,_selectoccurence)
                            
                        with col2:
                            _list=words[(words["label"]==str(_randomAuthor)) \
                            & (words["writer_id"]!="000")]["writer_id"].unique()
                            if len(_list)==0:
                                st.write("Pas d'occurence")
                            else:
                                _tmpauthor=str(np.random.choice(_list))
                                _FileRef=words[(words["label"]==_randomAuthor) \
                              & (words["writer_id"]==_tmpauthor)]['file'].iloc[0]
                                #XRef=[]
                                imageR=cv2.imread(str(_FileRef), cv2.IMREAD_GRAYSCALE)
                                #XRef.append(np.array(imageR))
                                st.write("Writer : " + _tmpauthor + "/ Text : " +  _randomAuthor )
                                fig, ax = plt.subplots(nrows=len(words[(words["label"]==_randomAuthor) \
                              & (words["writer_id"]==_tmpauthor)]['file']) \
                                 , ncols=2, figsize=(10,8*len(words[(words["label"]==_randomAuthor) \
                              & (words["writer_id"]==_tmpauthor)]['file'])),constrained_layout = True)
                                j=0
                                for i, axi in enumerate(ax.flat):
                                    #X=[]
                                    if i%2==0:
                                        axi.set_title("Reference image")                                    
                                    #st.image(_imageRef)
                                        axi.imshow(imageR,cmap='gray')

                                    else: 
                                        file =words[(words["label"]==_randomAuthor) \
                              & (words["writer_id"]==_tmpauthor)]['file'].iloc[j]
                                        axi.set_title("Compared image")
                                    #st.image(file)
                                        image=cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                                    #X.append(np.array(image))
                                        axi.imshow(image,cmap='gray')
                                        j+=1
                                st.pyplot()
            if selectSampling=="Distribution des mots":

                def _metricsWords():
                  ########moyenne de nombre de mots par size ########
                  _df=words[['label','writer_id','word_size','file']]
                  _dfc=_df.groupby(['label','word_size']).count().reset_index().rename(columns={'file':'count'})
                  _dfm=_dfc['count'].groupby(_dfc['word_size']).agg({'mean','std'}).reset_index().fillna(0)
                
                  ########moyenne de nombre d'auteurs par size ########
                  _df2=words[['label','writer_id','word_size']].drop_duplicates()
                  _dfc2=_df2.groupby(['label','word_size'])['writer_id'].count().reset_index().rename(columns={'writer_id':'count'})
                
                  _dfm2=_dfc2['count'].groupby(_dfc2['word_size']).agg({'mean','std'}).reset_index().fillna(0)
                
                
                
                  ######Graphes###
                  fig, (ax1, ax2) = plt.subplots(2, 1,constrained_layout = True)
                  fig = plt.figure(figsize=(30,20))
                
                  ax1.set_title("Mean Number of Words per Word Size")
                  ax1.set_ylabel('Mean number of words')
                  ax1.set_xlabel('Word Size')
                  x=_dfm['word_size']
                  y=_dfm['mean']
                  ax1.set_xticks(x)
                  err=_dfm['std']
                  ax1.scatter(x,y)
                  ax1.errorbar(x,y,yerr=err)
                
                  ax2.set_title("Mean Number of Words per Word Size")
                  ax2.set_ylabel('Mean number of words')
                  ax2.set_xlabel('Word Size')
                  x=_dfm2['word_size']
                  y=_dfm2['mean']
                  ax2.set_xticks(x)
                  err=_dfm2['std']
                  ax2.scatter(x,y)
                  ax2.errorbar(x,y,yerr=err)
                  plt.show()
                  st.pyplot()
                

                def _distriWperAuthor():
                  _df=words[['label','writer_id','word_size','file']]
                  _dfc=_df.groupby(['label','word_size','writer_id']).count().reset_index().rename(columns={'file':'count'})
                
                  _df=words[['label','word_size','file']]
                  _dft=_df.groupby(['label','word_size']).count().reset_index().rename(columns={'file':'total'})
                
                  _dfw0=pd.merge(_dfc,_dft[['total','label']], how='inner',on=['label'])
                  _dfw0['_percent']=round(100*_dfw0['count']/_dfw0['total'],2)
                  _txt=random.choice(_dfw0[(_dfw0['word_size']>2) & (_dfw0['count']>2) & (_dfw0['_percent']> 50) & (_dfw0['_percent'] < 100)]['label'].unique())
                
                  _print=words[words['label']==_txt][['file','writer_id','label']].sort_values(by=['writer_id'])
                  _tmp=_print.groupby(['writer_id']).count().reset_index().rename(columns={'file':'count'})[['writer_id','count']].\
                  sort_values(by=['count'],ascending=False)
                  writerMax=_tmp['writer_id'].iloc[0]
                  fig, ax = plt.subplots(nrows=max(len(_print[_print['writer_id']!=writerMax]),len(_print[_print['writer_id']==writerMax])) \
                                              , ncols=2, figsize=(10,8),constrained_layout = True)
                  linOth=0
                  linMax=0
                  for file,writer,label in zip(_print['file'],_print['writer_id'],_print['label']):
                    if writer == writerMax:
                      axi=ax[linMax][0]
                      axi.set_title("Writer : " + writer + " Label: " + label )                                   
                      image=cv2.imread(str(file), cv2.IMREAD_GRAYSCALE) 
                      axi.imshow(image,cmap='gray')
                      linMax+=1                         
                    else :
                      axi=ax[linOth][1]
                      axi.set_title("Writer : " + writer + " Label: " + label )                                    
                      image=cv2.imread(str(file), cv2.IMREAD_GRAYSCALE) 
                      axi.imshow(image,cmap='gray')
                      linOth+=1   
                  st.pyplot()
                  
                st.subheader("Distibution : Mots")
                _metricsWords()
                st.subheader("Exemple de haute contribution d'un auteur")                  
                if st.button("Refresh"):
                    _distriWperAuthor()          
                            
                            
                            

                
                           
                
                                
                
                

