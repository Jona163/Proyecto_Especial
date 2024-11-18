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

# Procesamiento del dataset
def process_and_visualize(data_path):
    """Carga, procesa el dataset y genera visualizaciones."""
    # Cargar el dataset
    df = load_kdd_dataset(data_path)

    # Generar estadísticas descriptivas
    stats = df.describe().to_html(classes="table table-striped", border=0)

    # Distribución de 'protocol_type'
    protocol_counts = df["protocol_type"].value_counts()
    protocol_plot_path = os.path.join(RESULTS_DIR, "protocol_distribution.png")
    protocol_counts.plot(kind="bar", color="skyblue")
    plt.title("Distribución de 'protocol_type'")
    plt.xlabel("Protocol Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(protocol_plot_path)
    plt.close()
