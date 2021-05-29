from .Gru import load_modelGru
from .Lstm import load_modelLstm
import random

def predictDktLstm(fileName):
	return load_modelLstm(fileName)
	
def predictDktGru(fileName):
	return load_modelGru(fileName)