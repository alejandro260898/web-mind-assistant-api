import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from keras_preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences

class ProcesadorTexto():
    FILTRO_RESPUESTA = '"#$%&()*+-/:;<=>@[\\]^`{|}~'
    TOKEN_DESCONOCIDO = '<UNK>'
    MODO_PADDING = 'post'
    RUTA_TOKENIZER_PREGUNTAS = '../data/tokenizer_preguntas.pkl'
    RUTA_TOKENIZER_RESPUESTAS = '../data/tokenizer_respuestas.pkl'
    
    def __init__(self, rutaDataset = ''):
        self.rutaDataset = rutaDataset
        self.tokenizer_pregunta = Tokenizer(oov_token=self.TOKEN_DESCONOCIDO)
        self.tokenizer_respuesta = Tokenizer(filters=self.FILTRO_RESPUESTA, oov_token=self.TOKEN_DESCONOCIDO)
        
    def cargarTokenizer(self):
        ruta_preguntas = Path(self.RUTA_TOKENIZER_PREGUNTAS)
        ruta_respuestas = Path(self.RUTA_TOKENIZER_RESPUESTAS)
        
        if(ruta_preguntas.is_file() and ruta_respuestas.is_file()):
            with open(self.RUTA_TOKENIZER_PREGUNTAS, 'rb') as file:
                self.tokenizer_pregunta = pickle.load(file)
            with open(self.RUTA_TOKENIZER_RESPUESTAS, 'rb') as file:
                self.tokenizer_respuesta = pickle.load(file)
            return True
        else:
            return False
        
    def cargar(self, colA = 'preguntas', colB = 'respuestas'):
        df = pd.read_excel(self.rutaDataset) 
        self.preguntas = df[colA].values
        self.respuestas = df[colB].values
        
        return self.preguntas, self.respuestas
        
    def guardar(self):
        self.tokenizer_pregunta.fit_on_texts(self.preguntas)
        self.tokenizer_respuesta.fit_on_texts(self.respuestas)

        with open(self.RUTA_TOKENIZER_PREGUNTAS, 'wb') as file:
            pickle.dump(self.tokenizer_pregunta, file)
        with open(self.RUTA_TOKENIZER_RESPUESTAS, 'wb') as file:
            pickle.dump(self.tokenizer_respuesta, file)
        
    def obtenerPreguntas(self):
        return self.preguntas
    
    def obtenerRespuestas(self):
        return self.respuestas
    
    def entrenar(self):
        self.secuencias_preguntas = self.tokenizer_pregunta.texts_to_sequences(self.preguntas)
        self.max_len_preguntas = max([len(seq) for seq in self.secuencias_preguntas])
        
        self.secuencias_respuestas = self.tokenizer_respuesta.texts_to_sequences(self.respuestas)
        self.max_len_respuestas = max([len(seq) for seq in self.secuencias_respuestas])
        
        self.max_len_vocabulario = max(self.max_len_preguntas, self.max_len_respuestas)
        
        self.secuencias_preguntas_padded = pad_sequences(
            self.secuencias_preguntas, 
            maxlen=self.max_len_vocabulario, 
            padding=self.MODO_PADDING
        )
        self.secuencias_respuestas_padded = pad_sequences(
            self.secuencias_respuestas, 
            maxlen=self.max_len_vocabulario, 
            padding=self.MODO_PADDING
        )
        

    def obtenerSecuencias(self, tipo:str = 'pregunta') -> np.ndarray:
        if(tipo == 'pregunta'): return self.secuencias_preguntas_padded
        else: return self.secuencias_respuestas_padded
    
    def obtenerMaxTamSecuencia(self, tipo:str = 'pregunta'):
        if(tipo == 'pregunta'): return self.max_len_preguntas
        elif(tipo == 'respuesta'): return self.max_len_respuestas
        else: return self.max_len_vocabulario
    
    def obtenerIndicesPalabras(self, tipo:str = 'pregunta'):
        if(tipo == 'pregunta'): return self.tokenizer_pregunta.index_word
        else: return self.tokenizer_respuesta.index_word
        
    def obtenerPalabrasIndices(self, tipo:str = 'pregunta'):
        if(tipo == 'pregunta'): return self.tokenizer_pregunta.word_index
        else: return self.tokenizer_respuesta.word_index
        
    def adaptarPregunta(self, pregunta:str = ''):
        secuencia = self.tokenizer_pregunta.texts_to_sequences([pregunta])
        return pad_sequences(secuencia, maxlen=self.max_len_vocabulario, padding=self.MODO_PADDING)