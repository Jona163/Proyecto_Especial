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
    
    # Verifica las distribuciones de clases
    print(f"Distribución de clases en el conjunto de entrenamiento: {train[stratify_col].value_counts()}")
    print(f"Distribución de clases en el conjunto de validación: {val[stratify_col].value_counts()}")
    print(f"Distribución de clases en el conjunto de prueba: {test[stratify_col].value_counts()}")
    
    return train, val, test

# Función de preprocesamiento
def preprocess_dataset(X):
    """Preprocesa el dataset manejando valores nulos."""
    print("Valores nulos antes del preprocesamiento:")
    print(X.isnull().sum())
    
    # Asignar NaN a valores específicos
    X.loc[(X["src_bytes"] > 400) & (X["src_bytes"] < 800), "src_bytes"] = np.nan
    X.loc[(X["src_bytes"] > 500) & (X["src_bytes"] < 2000), "src_bytes"] = np.nan

    # Manejo de valores nulos con imputación por mediana
    imputer = SimpleImputer(strategy="median")
    numeric_data = X.select_dtypes(exclude=["object"])
    numeric_data_imputed = imputer.fit_transform(numeric_data)
    X_imputed = pd.DataFrame(numeric_data_imputed, columns=numeric_data.columns, index=X.index)
    
    print("Valores nulos después del preprocesamiento:")
    print(X_imputed.isnull().sum())
    return X_imputed
