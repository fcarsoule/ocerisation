# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:31:24 2022

@author: frede
"""



from os import listdir
from os.path import isfile, join

import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
import imghdr




class Importation():
    def __init__(self, repertoireDB = "../database"):
        self.repertoireDB = repertoireDB
        self.repertoireXML = join(repertoireDB, "xml")
        self.repertoireAscii = join(repertoireDB, "ascii")
        self.df_forms = self.get_forms_ini()
        
   
    def get_forms_ini(self, file_name="forms.txt"):
       
        with open(join(self.repertoireAscii,file_name), 'r') as f:
            forms = f.readlines()


        df_forms = pd.DataFrame([s.split(" ") for s in forms if s[0]!="#"])
        df_forms.columns = ['form_id', 'writer_id', 'number_of_sentences',
                            'word_segmentation',
                            'number_of_lines', 'number_of_lines_correctly_segmented',
                            'number_of_words', 'number_of_words_correctly_segmented']
        
        df_forms["file"] = join(self.repertoireDB, "forms/")+ "/" + df_forms['form_id'] + ".png"
        return df_forms

    
    def get_xml(self):
        # Récupération du texte des pages
        fichiersXML = [f for f in listdir(self.repertoireXML) if isfile(join(self.repertoireXML, f))]
        fichiersXML.sort()
        
        df_forms_xml = pd.DataFrame(list(
            zip([f.split(".")[0] for f in fichiersXML],
                             [f for f in fichiersXML])
                                ),
                             columns=['form_id', 'file_name'])
        df_forms_xml["writer_id"] = ""
        df_forms_xml["text"] = ""
        
        for fichierXML in fichiersXML:
            tree = ET.parse(f"{self.repertoireXML}/{fichierXML}")
            form_id = fichierXML.split(".")[0]
            #print("Fichier : ", fichiersXML[0])
            root = tree.getroot()
            writer_id = root.attrib["writer-id"]
            for child in root:
                #print(child.tag)
                if child.tag=="machine-printed-part":
                    text = ""
                    for line in child:
                        text += line.attrib['text'] + "\n"
                        #print(line.attrib['text'])
                    text = text[:-1]
                df_forms_xml.loc[df_forms_xml["form_id"]==form_id,["writer_id","text"]]=[writer_id, text]
        return df_forms_xml
    
    def get_forms(self):
        df_forms_xml = self.get_xml()
        df_forms = self.df_forms.merge(on="form_id", right=df_forms_xml[["form_id", "text"]])
        return df_forms
    
    def get_words(self, file_name="words.txt", drop_err_seg=True):
        
        with open(join(self.repertoireAscii, file_name), 'r') as f:
            words = f.readlines()
        df_words = pd.DataFrame([s.split(" ", maxsplit=8)
                                 for s in words if s[0] != "#"])
        df_words.columns = ['word_id', 'result_w_seg', 'greylevel',
                          'x', 'y', 'w', 'h', 'grammatical_tag', 'label']
    
      #supprimer lignes avec erreur segmentation:
        if drop_err_seg:
            df_words.drop(index=df_words[df_words["result_w_seg"]=="err"].index, inplace=True)
       
        # Suppression des lignes dont les fichiers ne sont pas des images
        png_words_errors = [4152, 113621]
        df_words.drop(index=png_words_errors, inplace=True)
    
      # supprimer \n en fin ligne
        df_words["label"] = df_words["label"].str.strip()
    
      # reconsituer chemin de l'image - Ajout colonne form_id et writer_id
        df_file = df_words['word_id'].str.split(pat="-", n=2, expand=True)
        df_words['file'] = join(self.repertoireDB, "words/") + df_file[0] + "/" + \
          df_file[0] + "-" + df_file[1] + "/" + df_words['word_id'] + ".png"
        df_words['form_id'] = df_file[0] + '-' + df_file[1]

        df_words = df_words.merge(on="form_id", right=self.df_forms[["form_id", "writer_id"]])
        
        df_words["word_size"] = df_words.label.str.len()
        
      
        return df_words
  
    def get_lines(self, file_name="lines.txt", drop_err_seg=True):
        with open(join(self.repertoireAscii,file_name), 'r') as f:
            lines = f.readlines()
    
        df_lines = pd.DataFrame([s.split(" ", maxsplit=8) for s in lines if s[0]!="#"])
        df_lines.columns = ['line_id', 'result_w_seg', 'greylevel', 'number_of_components',
                            'x', 'y', 'w', 'h','label'] 
        
        #supprimer lignes avec erreur segmentation:
        if drop_err_seg:
            df_lines.drop(index=df_lines[df_lines["result_w_seg"]=="err"].index, inplace=True)
        
        #supprimer \n en fin ligne
        # df_lines["label"] = df_lines["label"].str.strip()
        labels = []
        for label in df_lines.label:
            labels.append(label.replace('|',' ').strip())
        df_lines.label = labels
        df_lines['file'] = join(self.repertoireDB, "lines/") +\
            df_lines['line_id'].str.split(pat="-", n=1, expand=True)[0] + "/" +\
            df_lines['line_id'].str.rsplit(pat="-", n=1, expand=True)[0] \
            + "/" + df_lines['line_id'] + ".png"
            
        df_lines['form_id'] = df_lines['line_id'].str.rsplit(pat="-", n=1, expand=True)[0]
        df_lines = df_lines.merge(on="form_id", right=self.df_forms[["form_id", "writer_id"]])
        
        return df_lines

    def get_sentences(self, file_name="sentences.txt", drop_err_seg=True):
        with open(join(self.repertoireAscii,file_name), 'r') as f:
            sentences = f.readlines()
    
        df_sentences = pd.DataFrame([s.split(" ", maxsplit=9) for s in sentences if s[0]!="#"])
        df_sentences.columns = ['sentence_id', 'sentence_number', 'result_w_seg',
                                'graylevel', 'number_of_components',
                                'x', 'y', 'w', 'h','label'] 
        #supprimer lignes avec erreur segmentation:
        if drop_err_seg:
            df_sentences.drop(index=df_sentences[df_sentences["result_w_seg"]=="err"].index, inplace=True)
        
        #supprimer \n en fin ligne
        df_sentences["label"] = df_sentences["label"].str.strip()
        
        df_sentences["line_number"] = df_sentences["sentence_id"].str.rsplit(pat="-",n=1, expand=True)[1]
        
        df_sentences["form_id"]= df_sentences["sentence_id"].str.rsplit(pat="-",n=2, expand=True)[0]
        df_sentences['file'] = join(self.repertoireDB, "sentences/") +\
            df_sentences['sentence_id'].str.split(pat="-", n=1, expand=True)[0] + "/" +\
            df_sentences['sentence_id'].str.rsplit(pat="-", n=2, expand=True)[0] \
            + "/" + df_sentences['sentence_id'] + ".png"
        #df_sentences['form_id'] = df_sentences['sentence_id'].str.rsplit(pat="-", n=2, expand=True)[0]    
        df_sentences = df_sentences.merge(on="form_id", right=self.df_forms[["form_id", "writer_id"]])
        
        return df_sentences
    


    def get_images_size(self, files):
        """
    
        Parameters
        ----------
        files : Series chemin des images
    
        Returns
        -------
        Series (width, height)
    
        """
        img_size = files.apply(lambda x: Image.open(x).size)
        return img_size
      
      

    def get_png_errors(self, df, col_name):
        """
        retourne les index des lignes qui ne sont pas des images
        Parameters
        ----------
        df : DataFrame
    
        col_name : Series
            Colonne contenant l'adresse de l'image
    
        Returns
        -------
        index des lignes des images en erreur
    
        """
        s = df[col_name].apply(lambda x: imghdr.what(x))
        return s[s.isna()].index



    def get_characters(self, words):
        """
        Parameters
        ----------
        words : Series
            Liste de str
    
        Returns
        -------
        max_len : int
            longueur du mot le plus long
        longest_word : str
            mot le plus long
        characters : set
            ensemble des caractêres
    
        """
        characters = set()
        max_len = 0
        longest_word = ""
        for word in words:
            if len(word)>max_len:
                max_len = len(word)
                longest_word = word
            for c in word:
                characters.add(c)
        return max_len, longest_word, characters




