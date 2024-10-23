import sys
import os
import numpy as np
from keras_preprocessing.sequence import pad_sequences

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

from preprocesamiento.ProcesadorTexto import ProcesadorTexto
from posprocesador.FastText import FastText
from model.LSTM import ModeloLSTM
from sklearn.model_selection import train_test_split

RUTA_DATASET = '../data/dataset.xlsx'

class ChatBot():
    EPOCAS = 300
    DIMENCION_EMBEDDING = 80
    UNIDADES = 250
    TAM_LOTE = 32
    VALIDACION = 0.2
    
    def __init__(self):
        self.fast_text = FastText()
        
    def cargarFastText(self):
        return
    
    def cargar(self):
        preprocesador = ProcesadorTexto(RUTA_DATASET)
        preguntas, respuesta = preprocesador.cargar('USUARIO', 'ASISTENTE')

        _, self.preguntas_test, _, self.respuesta_test = train_test_split(
            preguntas, 
            respuesta, 
            test_size=0.2, 
            random_state=42
        )
        
        if(not preprocesador.cargarTokenizer()): preprocesador.guardar()
        # else existe el archivo con los tokenizer correspondiente
        preprocesador.entrenar()
        tam_vocabulario_preguntas = len(preprocesador.obtenerIndicesPalabras('pregunta')) + 1 
        tam_vocabulario_respuestas = len(preprocesador.obtenerIndicesPalabras('respuesta')) + 1
        tam_max = preprocesador.obtenerMaxTamSecuencia('vocabulario') # palabras max. sin repetir entre preguntas y respuestas

        modeloLSTM = ModeloLSTM(self.DIMENCION_EMBEDDING, self.UNIDADES, self.TAM_LOTE, self.VALIDACION)
        if(not modeloLSTM.construirModelo(tam_vocabulario_preguntas, tam_max, tam_vocabulario_respuestas)):
            preguntas_train, _, respuesta_train, _ = train_test_split(
                preprocesador.obtenerSecuencias('pregunta'), 
                preprocesador.obtenerSecuencias('respuesta'), 
                test_size=0.2, 
                random_state=42
            )
            X = preguntas_train
            y = np.expand_dims(respuesta_train, axis=-1)
            modeloLSTM.entrenar(X, y, self.EPOCAS)
        # else existe un archivo con el modelo entrenado
        
        self.tam_max = tam_max
        self.modeloLSTM = modeloLSTM
        self.preprocesador = preprocesador
    
    def procesar(self, pregunta:str = ''):
        pregunta_pad = self.preprocesador.adaptarPregunta(pregunta)
        prediccion = self.modeloLSTM.predeccir(pregunta_pad)
        
        respuesta = [self.preprocesador.obtenerIndicesPalabras('respuesta').get(i, '') for i in prediccion if i > 0]
        respuesta = ' '.join(respuesta)
        return respuesta
        
    def evaluar(self):
        for pregunta in self.preguntas_test:
            print(self.procesar(pregunta))
            print('\n')

chat_bot = ChatBot()
chat_bot.cargar()
chat_bot.evaluar()