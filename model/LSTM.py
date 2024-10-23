import numpy as np
from pathlib import Path
from keras.api.models import Sequential, load_model
from keras.api.layers import Embedding, LSTM, Dense, TimeDistributed

class ModeloLSTM():
    DIMENCION_EMBEDDING = 100
    UNIDADES = 128
    TAM_LOTE = 32
    VALIDACION = 0.2
    NOM_ARCHIVO = './modelo_lstm.h5'
    
    def __init__(self, dim_embedding = 100, unidades = 128, tam_lote = 32, validacion = 0.2):
        self.DIMENCION_EMBEDDING = dim_embedding
        self.UNIDADES = unidades
        self.TAM_LOTE = tam_lote
        self.VALIDACION = validacion
    
    def construirModelo(self, tam_vocabulario_entrada = 0, max_tam_entrada = 0, tam_vocabulario_salida = 0):
        ruta = Path(self.NOM_ARCHIVO)
        if(ruta.is_file()):
            self.model = load_model(self.NOM_ARCHIVO)
            return True
        else:
            self.model = Sequential()
            self.model.add(
                Embedding(
                    input_dim=tam_vocabulario_entrada, 
                    output_dim=self.DIMENCION_EMBEDDING, 
                    input_length=max_tam_entrada
                )
            )
            self.model.add(LSTM(self.UNIDADES, return_sequences=True))
            self.model.add(
                TimeDistributed(
                    Dense(tam_vocabulario_salida, activation='softmax')
                )
            )
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model.summary()
            return False
        
    def entrenar(self, X:np.ndarray, y:np.ndarray, epocas:int = 50):
        res = self.model.fit(
            X, y, 
            batch_size=self.TAM_LOTE, 
            epochs=epocas, 
            validation_split=self.VALIDACION
        )
        self.model.save(self.NOM_ARCHIVO)
        
    def predeccir(self, x:str):
        prediccion = self.model.predict(x)
        return np.argmax(prediccion, axis=-1)[0]