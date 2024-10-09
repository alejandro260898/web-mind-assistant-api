import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '')))

from preprocesamiento.ProcesadorTexto import ProcesadorTexto
from model.LSTM import ModeloLSTM

RUTA_DATASET = '../data/dataset.xlsx'

preprocesador = ProcesadorTexto(RUTA_DATASET)
preprocesador.cargar('USUARIO', 'ASISTENTE')
preprocesador.entrenarPreguntas()
preprocesador.entrenarRespuesta()