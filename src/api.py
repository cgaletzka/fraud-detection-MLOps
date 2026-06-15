import sys
sys.path.append("src")
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from utils import load_config, load_model, encode_vars

# these run once when the server starts
app = FastAPI()
config = load_config("config.yaml")
model = load_model("models/rf_model.pkl")

class Claim(BaseModel):
    Month: str
    WeekOfMonth: int
    DayOfWeek: str
    Make: str
    AccidentArea: int
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int
    Sex: int
    MaritalStatus: str
    Age: int
    Fault: int
    PolicyType: str
    VehiclePrice: int
    DriverRating: int
    PastNumberOfClaims: int
    AgeOfVehicle: int
    PoliceReportFiled: int
    AgentType: int
    NumberOfSuppliments: int
    AddresChange_Claim: int
    NumberOfCars: int
    Year: int

feature_columns = load_model("models/feature_columns.pkl")

# predict new data
@app.post("/predict")
def predict(claim: Claim):
    df = pd.DataFrame([claim.model_dump()])
    df_enc = encode_vars(df, config)
    
    # add missing columns with 0, reorder to match training
    df_enc = df_enc.reindex(columns=feature_columns, fill_value=0)
    
    y_proba = model.predict_proba(df_enc)[:, 1][0]
    threshold = 0.5
    y_pred = int(y_proba >= threshold)
    return {"fraud_probability": float(y_proba), "fraud_predicted": y_pred}

# only needed if running directly with python src/api.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)