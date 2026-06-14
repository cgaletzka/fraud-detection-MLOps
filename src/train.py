import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import load_data, load_config, encode_vars, split_data

def run_grid_search(X_train, y_train, config):
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    mlflow.sklearn.autolog()
    with mlflow.start_run():      
        # set-up grid search
        grid = GridSearchCV(
            estimator=RandomForestClassifier(**config['model']['RF']['params']),
            param_grid=config['model']['RF']['param_grid'],
            scoring=['recall', 'precision', 'f1_macro', 'roc_auc'],
            refit='roc_auc',
            cv=5,
            return_train_score=True)
        # run grid search
        grid.fit(X_train, y_train)
    return grid

if __name__ == "__main__":
    df = load_data("data/processed/carclaims_processed.csv")
    config = load_config("config.yaml")
    df_enc = encode_vars(df, config)
    
    X_train, X_test, y_train, y_test = split_data(df_enc, config)
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl") # save feature column for API later
    
    rf_grid = run_grid_search(X_train, y_train, config)
    joblib.dump(rf_grid.best_estimator_, "models/rf_model.pkl")
    
    print("Best parameters:", rf_grid.best_params_)
    print("Best recall:", rf_grid.cv_results_['mean_test_recall'][rf_grid.best_index_])
    print("Best precision:", rf_grid.cv_results_['mean_test_precision'][rf_grid.best_index_])
    print("Best F1-macro:", rf_grid.cv_results_['mean_test_f1_macro'][rf_grid.best_index_])
    print("Best ROC-AUC:", rf_grid.cv_results_['mean_test_roc_auc'][rf_grid.best_index_])

    