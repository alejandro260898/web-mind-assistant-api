import fasttext
import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Embedding

# Dataset pequeño de ejemplo
preguntas = ["Hola, ¿cómo estás?", "¿Qué es inteligencia artificial?", "Dame un consejo para relajarme"]
respuestas = ["Estoy bien, gracias", "La inteligencia artificial es la simulación de procesos humanos por máquinas", "Te recomiendo practicar la meditación"]

# Tokenizers separados para preguntas y respuestas
tokenizer_preguntas = Tokenizer()
tokenizer_respuestas = Tokenizer()

tokenizer_preguntas.fit_on_texts(preguntas)
tokenizer_respuestas.fit_on_texts(respuestas)

sequences_preguntas = tokenizer_preguntas.texts_to_sequences(preguntas)
sequences_respuestas = tokenizer_respuestas.texts_to_sequences(respuestas)

# Padding para igualar la longitud
max_len_preg = max([len(x) for x in sequences_preguntas])
max_len_resp = max([len(x) for x in sequences_respuestas])

X = pad_sequences(sequences_preguntas, maxlen=max_len_preg, padding='post')
y = pad_sequences(sequences_respuestas, maxlen=max_len_resp, padding='post')

# Paso 3: Entrenar FastText para preguntas y respuestas por separado
# Guardar las preguntas y respuestas en archivos de texto para que FastText las entrene
with open("preguntas.txt", "w", encoding='utf-8') as f:
    for preg in preguntas:
        f.write(preg + "\n")

with open("respuestas.txt", "w", encoding='utf-8') as f:
    for resp in respuestas:
        f.write(resp + "\n")

# Entrenar modelos FastText
modelo_fasttext_preguntas = fasttext.train_unsupervised("preguntas.txt", model='skipgram', minCount=1)
modelo_fasttext_respuestas = fasttext.train_unsupervised("respuestas.txt", model='skipgram', minCount=1)

# Paso 4: Crear embeddings para preguntas y respuestas
def crear_embedding_matrix(tokenizer, fasttext_model, vocab_size):
    embedding_dim = fasttext_model.get_dimension()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in tokenizer.word_index.items():
        if idx < vocab_size:
            embedding_vector = fasttext_model.get_word_vector(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
    
    return embedding_matrix

# Crear matrices de embeddings
vocab_size_preg = len(tokenizer_preguntas.word_index) + 1
vocab_size_resp = len(tokenizer_respuestas.word_index) + 1

embedding_matrix_preguntas = crear_embedding_matrix(tokenizer_preguntas, modelo_fasttext_preguntas, vocab_size_preg)
embedding_matrix_respuestas = crear_embedding_matrix(tokenizer_respuestas, modelo_fasttext_respuestas, vocab_size_resp)

# Paso 5: Construir el modelo LSTM
embedding_dim = modelo_fasttext_preguntas.get_dimension()

model = Sequential()
# Embedding para preguntas
model.add(Embedding(input_dim=vocab_size_preg, 
                    output_dim=embedding_dim, 
                    weights=[embedding_matrix_preguntas], 
                    input_length=max_len_preg, 
                    trainable=False))
model.add(LSTM(64, return_sequences=False))
# Capa densa para salida
model.add(Dense(vocab_size_resp, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Ajustar el formato de las etiquetas (no deben estar expandidas)
y = np.argmax(y, axis=-1)  # Cada valor en 'y' debe ser un índice de la palabra

# Entrenar el modelo
model.fit(X, y, epochs=100, verbose=2)

# Paso 6: Función para generar respuestas creativas
def generar_respuesta_creativa(input_text):
    seq = tokenizer_preguntas.texts_to_sequences([input_text])
    padded_seq = pad_sequences(seq, maxlen=max_len_preg, padding='post')
    
    pred = model.predict(padded_seq)
    idx = np.argmax(pred[0])

    for word, word_idx in tokenizer_respuestas.word_index.items():
        if word_idx == idx:
            return word

# Probar el chatbot
input_usuario = "Hola, ¿cómo estás?"
respuesta_creativa = generar_respuesta_creativa(input_usuario)
print("Chatbot:", respuesta_creativa)