import os
import pandas as pd
import numpy as np
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score,
    recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

# Configuración de directorios
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para cargar el dataset NSL-KDD
def load_kdd_dataset(data_path, sample_fraction=0.1):
    """Carga y reduce el tamaño del conjunto de datos."""
    with open(data_path, 'r') as file:
        dataset = arff.load(file)
    attributes = [attr[0] for attr in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=attributes)
    return df.sample(frac=sample_fraction, random_state=42)  # Reduce el tamaño

# Procesamiento y evaluación
def logistic_regression_pipeline(data_path):
    # Cargar el dataset reducido
    df = load_kdd_dataset(data_path)
    
    # División en conjuntos
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df["class"])
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42, stratify=test_set["class"])
