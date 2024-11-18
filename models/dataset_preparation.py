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

# Función para dividir el conjunto de datos
def split_dataset(df, stratify_col=None, test_size=0.4, val_size=0.5, random_state=42):
    """Divide el conjunto de datos en entrenamiento, validación y prueba."""
    stratify = df[stratify_col] if stratify_col else None
    train, temp = train_test_split(df, test_size=test_size, stratify=stratify, random_state=random_state)
    stratify_temp = temp[stratify_col] if stratify_col else None
    val, test = train_test_split(temp, test_size=val_size, stratify=stratify_temp, random_state=random_state)
