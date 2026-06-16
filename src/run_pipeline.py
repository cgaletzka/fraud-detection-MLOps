from utils import load_data, save_data, load_config
from train import train_model
from prepare_data import clean_data


if __name__ == "__main__":
    
    # load data
    df = load_data("data/raw/carclaims.csv")
    config = load_config("config.yaml")

    # preprocessing and save
    df_processed = clean_data(df)
    save_data(df_processed, "data/processed/carclaims_processed.csv")

    # train model
    train_model(df_processed, config)