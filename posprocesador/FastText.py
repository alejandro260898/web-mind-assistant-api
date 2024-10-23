import fasttext

class FastText:
    RUTA_ARCHIVO_PREGUNTAS = './preguntas.txt'
    RUTA_ARCHIVO_RESPUESTAS = './respuestas.txt'
    
    def __init__(self):
        pass
    
    def entrenar(self, data = [], tipo = 'pregunta'):
        rutaArchivo = (tipo == 'pregunta') if self.RUTA_ARCHIVO_PREGUNTAS else self.RUTA_ARCHIVO_RESPUESTAS
        
        with open(rutaArchivo, 'w') as f:
            for d in data: f.write(f'{d}\n')
            
        modelo_fasttext_preguntas = fasttext.train_unsupervised('preguntas.txt', model='skipgram')
    
    def generarRespuestaCreativa(self, respuestaOrg:str = ''):
        return