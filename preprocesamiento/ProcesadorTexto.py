import pandas as pd
from keras_preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences

class ProcesadorTexto():
    MODO_PADDING = 'post'
    
    def __init__(self, rutaDataset = ''):
        self.rutaDataset = rutaDataset
        self.tokenizer_pregunta = Tokenizer()
        self.tokenizer_respuesta = Tokenizer()
        
    def cargar(self, colA = 'preguntas', colB = 'respuestas'):
        df = pd.read_excel(self.rutaDataset)
        
        self.preguntas = df[colA].values
        self.respuestas = df[colB].values
        
    def obtenerPreguntas(self):
        return self.preguntas
    
    def obtenerRespuestas(self):
        return self.respuestas
    
    def entrenarPreguntas(self):
        self.tokenizer_pregunta.fit_on_texts(self.preguntas)
        self.secuencias_preguntas = self.tokenizer_pregunta.texts_to_sequences(self.preguntas)
        self.maxlen_preguntas = max([len(seq) for seq in self.secuencias_preguntas])
        self.secuencias_preguntas_padded = pad_sequences(self.secuencias_preguntas, maxlen=self.maxlen_preguntas, padding=self.MODO_PADDING)
        
    def entrenarRespuesta(self):
        self.tokenizer_respuesta.fit_on_texts(self.respuestas)
        self.secuencias_respuestas = self.tokenizer_respuesta.texts_to_sequences(self.respuestas)
        self.maxlen_respuestas = max([len(seq) for seq in self.secuencias_respuestas])
        self.secuencias_respuestas_padded = pad_sequences(self.secuencias_respuestas, maxlen=self.maxlen_respuestas, padding=self.MODO_PADDING)

    def obtenerSecuenciasPreguntas(self):
        return self.secuencias_preguntas_padded
    
    def obtenerSecuenciasRespuestas(self):
        return self.secuencias_respuestas_padded