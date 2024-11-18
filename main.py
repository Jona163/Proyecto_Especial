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

