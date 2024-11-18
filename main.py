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
