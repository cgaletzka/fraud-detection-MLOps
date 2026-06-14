import joblib
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from utils import load_data, load_config, encode_vars, split_data

def load_model(path):
    return joblib.load(path)

def predict_fraud(X_test, y_test, model, config):
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        y_proba_fraud = model.predict_proba(X_test)[:, 1]
        threshold=0.5
        y_pred = (y_proba_fraud >= threshold).astype(int)
        
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba_fraud))
        mlflow.log_param("threshold", threshold)

    return 

if __name__ == "__main__":
    df = load_data("data/processed/carclaims_processed.csv")
    config = load_config("config.yaml")
    model = load_model("models/rf_model.pkl")
    df_enc = encode_vars(df, config)
    X_train, X_test, y_train, y_test = split_data(df_enc, config)
    predict_fraud(X_test, y_test, model, config)
   