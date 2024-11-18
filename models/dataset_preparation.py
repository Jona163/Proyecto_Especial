import os
import pandas as pd
import numpy as np
import arff
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Configuración de directorios
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para cargar el dataset NSL-KDD
def load_kdd_dataset(data_path):
    """Lectura del conjunto de datos NSL-KDD."""
    with open(data_path, 'r') as file:
        dataset = arff.load(file)
    df = pd.DataFrame(dataset["data"], columns=[attr[0] for attr in dataset["attributes"]])
    # Verifica si los datos se cargaron correctamente
    print("Primeras filas del dataset:")
    print(df.head())
    return df
