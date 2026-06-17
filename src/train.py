import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve

def train_model(df, config):
    
    y = df[config['data']['y']]
    X = df.drop(columns=[config['data']['y']])

    # get lists of the numerical, categorical, and binary columns
    int_columns = X.select_dtypes(include=["int64"]).columns

    numerical_cols = []
    binary_cols = []

    for col in int_columns:
        # If it has 2 or fewer unique values, it's binary
        if X[col].nunique() <= 2:
            binary_cols.append(col)
        else:
            numerical_cols.append(col)

    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', 'passthrough', numerical_cols), # no need for scaling for tree models
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('binary', 'passthrough', binary_cols)
    ])


    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(**config['model']['RF']['best_params']))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['train']['test_size'], random_state=config['train']['random_state'], stratify=y)

    # start mlflow logging
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= config['train']['threshold']).astype(int)
        
        test_scores = {
            "test_recall": recall_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_pred_proba)}

        # log all scores to mlflow
        mlflow.log_metrics(test_scores)
        mlflow.log_param('threshold', config['train']['threshold'])
        mlflow.sklearn.log_model(pipeline, "pipeline")

        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # plot precision and recall across thresholds
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, precisions[:-1], "g-", label="Precision", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "r-", label="Recall", linewidth=2)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.title("Precision and recall vs. threshold")
        plt.savefig("reports/precision_recall_curve.png")
        mlflow.log_artifact("reports/precision_recall_curve.png")

        print("Model training finished. Precision: ", test_scores['test_precision'], "Recall: ", test_scores['test_recall'])
    
    joblib.dump(pipeline, "models/rf_pipeline.pkl")
    
    # save feature columns
    feature_cols = pipeline[:-1].get_feature_names_out()
    joblib.dump(list(feature_cols), "models/feature_columns.pkl")