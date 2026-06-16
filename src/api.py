import sys
sys.path.append("src")
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from utils import load_config, load_model
from prepare_data import clean_data

# these run once when the server starts
app = FastAPI()
config = load_config("config.yaml")
pipeline = load_model("models/rf_pipeline.pkl")

# create a class with variables
class Claim(BaseModel):
    Month: str
    WeekOfMonth: int
    DayOfWeek: str
    Make: str
    AccidentArea: str
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int
    Sex: str
    MaritalStatus: str
    Age: int
    Fault: str
    PolicyType: str
    VehicleCategory: str
    VehiclePrice: str
    PolicyNumber: int
    RepNumber: int
    Deductible: int
    DriverRating: int
    Days_Policy_Accident: str
    Days_Policy_Claim: str
    PastNumberOfClaims: str
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    NumberOfSuppliments: str
    AddressChange_Claim: str
    NumberOfCars: str
    Year: int
    BasePolicy: str

feature_columns = load_model("models/feature_columns.pkl")

# predict new data
@app.post("/predict")
def predict(claim: Claim):
    df = pd.DataFrame([claim.model_dump()])
    df_processed = clean_data(df)
    
    # add missing columns with 0, reorder to match training
    #df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)
    
    y_pred_proba = pipeline.predict_proba(df_processed)[:, 1][0]
    y_pred = int(y_pred_proba >= config['train']['threshold'])

    return {"fraud_probability": float(y_pred_proba), "fraud_predicted": y_pred}

# only needed if running directly with python src/api.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)