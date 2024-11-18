import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import arff

# Configuración de directorios
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para cargar el dataset
def load_kdd_dataset(data_path):
    """Carga y devuelve el dataset NSL-KDD como un DataFrame."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset['attributes']]
    return pd.DataFrame(dataset['data'], columns=attributes)
