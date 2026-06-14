import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def encode_vars(df, config):
    return pd.get_dummies(df, columns=config['data']['cat'], drop_first=True)

def split_data(df, config):
    y = df[config['data']['y']]
    X = df.drop(columns=[config['data']['y']])  
    return train_test_split(X, y, test_size=config['split']['test_size'], random_state=config['split']['random_state'])