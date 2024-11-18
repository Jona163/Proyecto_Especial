from flask import Flask, render_template, request
import os
import importlib
import pandas as pd
import joblib
from flask import render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), "models")

def load_model(module_name):
    try:
        module_path = f"models.{module_name}"
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None

@app.route("/")
def index():
    return render_template("index.html")


def process_model(model):
    try:
        result = model.run()  
        return result.get("original_stats", ""), result.get("transformed_stats", ""), result.get("scatter_plot", ""), None
    except Exception as e:
        return None, None, None, f"Error durante el procesamiento: {str(e)}"

@app.route("/transformers", methods=["GET", "POST"])
def transformers():
    model = load_model("transformer_pipeline") 
    if model and hasattr(model, "run"):
        try:
            result = model.run()  
            original_stats = result.get("original_stats", "")
            transformed_stats = result.get("transformed_stats", "")
            scatter_plot = result.get("scatter_plot", "")
            error_message = None  
        except Exception as e:
            error_message = f"Error durante el procesamiento: {str(e)}"
            original_stats = transformed_stats = scatter_plot = None
    else:
        original_stats = transformed_stats = scatter_plot = None
        error_message = "Error: No se encontró el método 'run' en el modelo de transformadores."


    return render_template(
        "transformers.html", 
        original_stats=original_stats,
        transformed_stats=transformed_stats,
        scatter_plot=scatter_plot,
        error_message=error_message 
    )


@app.route("/evaluation", methods=["GET", "POST"])
def evaluation():
    model = load_model("evaluation_model")
    if model and hasattr(model, "evaluate"):
        result = model.evaluate()

        print("Resultado de evaluate:", result)

        precision = result.get("precision") if result.get("precision") else "No disponible"
        recall = result.get("recall") if result.get("recall") else "No disponible"
        f1 = result.get("f1") if result.get("f1") else "No disponible"
        confusion_matrix = result.get("confusion_matrix") if result.get("confusion_matrix") else "No disponible"
        roc_curve = result.get("roc_curve") if result.get("roc_curve") else "No disponible"
        precision_recall_curve = result.get("precision_recall_curve") if result.get("precision_recall_curve") else "No disponible"

        print("Métricas: Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))

    else:
        result = {}  
        precision = recall = f1 = "No disponible"
        confusion_matrix = roc_curve = precision_recall_curve = "No disponible"
