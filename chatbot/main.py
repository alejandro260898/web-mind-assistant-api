import sys
import os
import numpy as np
from keras_preprocessing.sequence import pad_sequences

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

from preprocesamiento.ProcesadorTexto import ProcesadorTexto
from posprocesador.FastText import FastText
from model.LSTM import ModeloLSTM
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import fasttext

RUTA_DATASET = '../data/dataset.xlsx'

class ChatBot():
    EPOCAS = 1500
    FACTOR_APRENDIZAJE = 0.001
    DIMENSION_EMBEDDING = 100
    UNIDADES = 256
    TAM_LOTE = 32
    VALIDACION = 0.2
    TAM_PRUEBA = 0.2
    ESTADO_ALEATORIO = 42
    
    def __init__(self):
        self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def cargarFastText(self, preguntas, respuestas, preprocesador):
        fast_text = FastText(preprocesador=preprocesador, tam_embedding=self.DIMENSION_EMBEDDING)
        
        with open(fast_text.RUTA_ARCHIVO_PREGUNTAS, 'w', encoding='utf-8') as f:
            for d in preguntas: f.write(f'{d}\n')
        with open(fast_text.RUTA_ARCHIVO_RESPUESTAS, 'w', encoding='utf-8') as f:
            for d in respuestas: f.write(f'{d}\n')
            
        modelo_preguntas = fasttext.train_unsupervised(input=fast_text.RUTA_ARCHIVO_PREGUNTAS, model='skipgram', dim=self.DIMENSION_EMBEDDING)
        modelo_respuestas = fasttext.train_unsupervised(input=fast_text.RUTA_ARCHIVO_RESPUESTAS, model='skipgram', dim=self.DIMENSION_EMBEDDING)
        
        fast_text.fijaModelo(modelo_preguntas, 'pregunta')
        fast_text.fijaModelo(modelo_respuestas, 'respuesta')
        fast_text.crearEmbedding('pregunta')
        fast_text.crearEmbedding('respuesta')
        return fast_text
    
    def leerDatos(self):
        self.preprocesador = ProcesadorTexto(RUTA_DATASET)
        preguntas, respuestas = self.preprocesador.cargar('USUARIO', 'ASISTENTE')

        _, self.preguntas_test, _, self.respuesta_test = train_test_split(
            preguntas, 
            respuestas, 
            test_size=self.TAM_PRUEBA, 
            random_state=self.ESTADO_ALEATORIO
        )
        return preguntas, respuestas
    
    def cargar(self):
        preguntas, respuestas = self.leerDatos()
        
        if(not self.preprocesador.cargarTokenizer()): self.preprocesador.guardar()
        # else existe el archivo con los tokenizer correspondiente
        self.preprocesador.entrenar()
        tam_vocabulario_preguntas = len(self.preprocesador.obtenerIndicesPalabras('pregunta')) + 1 
        tam_vocabulario_respuestas = len(self.preprocesador.obtenerIndicesPalabras('respuesta')) + 1
        tam_max = self.preprocesador.obtenerMaxTamSecuencia('vocabulario') # palabras max. sin repetir entre preguntas y respuestas
        
        self.fast_text = self.cargarFastText(preguntas, respuestas, self.preprocesador)

        self.modeloLSTM = ModeloLSTM(
            factor_aprendizaje=self.FACTOR_APRENDIZAJE,
            dim_embedding=self.DIMENSION_EMBEDDING, 
            unidades=self.UNIDADES, 
            tam_lote=self.TAM_LOTE, 
            validacion=self.VALIDACION,
            fasttext=self.fast_text
        )
        if(
            not self.modeloLSTM.construirModelo(
                tam_vocabulario_preguntas, 
                tam_max, 
                tam_vocabulario_respuestas
            )
        ):
            X = self.preprocesador.obtenerSecuencias('pregunta')
            y = np.expand_dims(self.preprocesador.obtenerSecuencias('respuesta'), axis=-1)
            self.modeloLSTM.entrenar(X, y, self.EPOCAS)
        # else existe un archivo con el modelo entrenado
        
        self.respuestas = respuestas
    
    def procesar(self, pregunta:str = ''):
        pregunta_pad = self.preprocesador.adaptarPregunta(pregunta)
        prediccion = self.modeloLSTM.predeccir(pregunta_pad)
        
        respuesta_base = [self.preprocesador.obtenerIndicesPalabras('respuesta').get(i, '') for i in prediccion if i > 0]
        respuesta_base = ' '.join(respuesta_base)
        
        prediccion_embedding = self.sbert_model.encode(respuesta_base, convert_to_tensor=True)
        respuestas_embeddings = self.sbert_model.encode(self.respuestas, convert_to_tensor=True)
        similitudes = util.cos_sim(prediccion_embedding, respuestas_embeddings)
        mejor_respuesta_idx = similitudes.argmax()
        mejor_respuesta = self.respuestas[mejor_respuesta_idx]
        
        return mejor_respuesta
        
    def evaluar(self):
        for pregunta in self.preguntas_test:
            print(self.procesar(pregunta))
            print('\n')
        
# chat_bot = ChatBot()
# chat_bot.cargar()