import fasttext
import numpy as np
from preprocesamiento.ProcesadorTexto import ProcesadorTexto

class FastText:
    RUTA_ARCHIVO_PREGUNTAS = '../data/preguntas.txt'
    RUTA_ARCHIVO_RESPUESTAS = '../data/respuestas.txt'
    
    def __init__(self, preprocesador:ProcesadorTexto, tam_embedding:int = 100):
        self.preprocesador = preprocesador
        self.TAM_EMBEDDING = tam_embedding
        
    def fijaModelo(self, modelo:fasttext, tipo = 'pregunta'):
        if(tipo == 'pregunta'): self.modelo_preguntas = modelo
        else: self.modelo_respuestas = modelo
            
    def crearEmbedding(self, tipo = 'pregunta'):
        palabras_indices = self.preprocesador.obtenerPalabrasIndices(tipo)
        tam_vocabulario = len(palabras_indices) + 1
        matriz = np.zeros((tam_vocabulario, self.TAM_EMBEDDING))

        if(tipo == 'pregunta'): modelo = self.modelo_preguntas 
        else: modelo = self.modelo_respuestas
    
        for palabra, i in palabras_indices.items():
            vector_embedding = modelo.get_word_vector(palabra)
            if vector_embedding is not None: matriz[i] = vector_embedding
                
        if(tipo == 'pregunta'):
            self.matriz_embedding_preguntas = matriz
        else:
            self.matriz_embedding_respuestas = matriz
            
    def obtenerMatrizEmbedding(self, tipo:str = 'pregunta'):
        if(tipo == 'pregunta'):
            return self.matriz_embedding_preguntas
        else:
            return self.matriz_embedding_respuestas
    
    def generarRespuestaCreativa(self, respuestaOrg:str = ''):
        return