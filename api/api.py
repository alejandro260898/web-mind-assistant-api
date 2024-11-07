import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

from chatbot.main import ChatBot
from flask import Flask, jsonify, request

app = Flask(__name__)

modelo = None

@app.route('/')
def inicio():
    return 'Esta funcionando...'

@app.route('/pregunta', methods=['POST'])
def darPregunta():
    pregunta = request.get_json()
    pregunta = pregunta.get('message')
    respuesta = modelo.procesar(pregunta)
    print(respuesta)
    return jsonify({ 
        'respuesta': respuesta 
    })

if __name__ == '__main__':
    modelo = ChatBot()
    modelo.cargar()
    app.run(host='0.0.0.0', port=5000)
