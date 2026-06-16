# Fraud detection using the car claims dataset

## What I did:

- Ran an exploratory data analysis including parameter optimization using optuna
- Created a pipeline for training a random forest classifier with mlflow
- Included API support using fastAPI and pydantic
- Included a web interface for easy predictions with gradio (https://b6b642f532ddb359bd.gradio.live/)

## Data:

- The data comes from an open insurance fraud dataset which can be downloaded from here: https://github.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection

## Approach

- The dataset is highly imbalanced with less than 4% of cases representing fraud. Using balanced training and threshold=0.5, the modell achieved ~0.91 recall at a precision of ~0.13 . Depending on the operational costs of manual checks, raising the treshold makes sense but will reduce recall and precision to ~0.25.

## Next steps:

- Feature engineering
- Balancing data with SMOTE and applying ensemble learning as described here: https://www.nature.com/articles/s41598-025-25700-2#Sec15
- Using docker to containerize the repository