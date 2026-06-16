import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def save_data(df, path):
    df.to_csv(path, index=False)

def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def load_model(path):
    return joblib.load(path)