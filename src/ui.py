import sys
sys.path.append("src")
import gradio as gr
import pandas as pd
from utils import load_config, load_model
from prepare_data import clean_data

# load model and config once
config = load_config("config.yaml")
pipeline = load_model("models/rf_pipeline.pkl")

def predict_fraud(Month, WeekOfMonth, DayOfWeek, Make, AccidentArea,
                  DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed,
                  Sex, MaritalStatus, Age, Fault, PolicyType, VehicleCategory,
                  VehiclePrice, PolicyNumber, RepNumber, Deductible, DriverRating,
                  Days_Policy_Accident, Days_Policy_Claim, PastNumberOfClaims,
                  AgeOfVehicle, AgeOfPolicyHolder, PoliceReportFiled,
                  WitnessPresent, AgentType, NumberOfSuppliments,
                  AddressChange_Claim, NumberOfCars, Year, BasePolicy):
    
    # put inputs into a dataframe
    data = {
        "Month": Month,
        "WeekOfMonth": WeekOfMonth,
        "DayOfWeek": DayOfWeek,
        "Make": Make,
        "AccidentArea": AccidentArea,
        "DayOfWeekClaimed": DayOfWeekClaimed,
        "MonthClaimed": MonthClaimed,
        "WeekOfMonthClaimed": WeekOfMonthClaimed,
        "Sex": Sex,
        "MaritalStatus": MaritalStatus,
        "Age": Age,
        "Fault": Fault,
        "PolicyType": PolicyType,
        "VehicleCategory": VehicleCategory,
        "VehiclePrice": VehiclePrice,
        "PolicyNumber": PolicyNumber,
        "RepNumber": RepNumber,
        "Deductible": Deductible,
        "DriverRating": DriverRating,
        "Days_Policy_Accident": Days_Policy_Accident,
        "Days_Policy_Claim": Days_Policy_Claim,
        "PastNumberOfClaims": PastNumberOfClaims,
        "AgeOfVehicle": AgeOfVehicle,
        "AgeOfPolicyHolder": AgeOfPolicyHolder,
        "PoliceReportFiled": PoliceReportFiled,
        "WitnessPresent": WitnessPresent,
        "AgentType": AgentType,
        "NumberOfSuppliments": NumberOfSuppliments,
        "AddressChange_Claim": AddressChange_Claim,
        "NumberOfCars": NumberOfCars,
        "Year": Year,
        "BasePolicy": BasePolicy
        }
    
    df = pd.DataFrame([data])
    df_processed = clean_data(df)
    y_proba = pipeline.predict_proba(df_processed)[:, 1][0]
    
    return f"Fraud probability: {y_proba:.1%}"

demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Dropdown(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], label="Month"),
        gr.Slider(1, 5, step=1, label="Week of Month"),
        gr.Dropdown(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Day of Week"),
        gr.Dropdown(["Honda", "Toyota", "Ford", "Mazda", "Chevrolet", "Pontiac", "Accura", "Dodge", "Mercury", "Jaguar", "Nisson", "VW", "Saab", "Saturn", "Porche", "BMW", "Mecedes", "Ferrari", "Lexus"], label="Make"),
        gr.Radio(["Urban", "Rural"], label="Accident Area"),
        gr.Dropdown(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Day of Week Claimed"),
        gr.Dropdown(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], label="Month Claimed"),
        gr.Slider(1, 5, step=1, label="Week of Month Claimed"),
        gr.Radio(["Female", "Male"], label="Sex"),
        gr.Dropdown(["Single", "Married", "Widow", "Divorced"], label="Marital Status"),
        gr.Slider(0, 99, step=1, label="Age"),
        gr.Radio(["Policy Holder", "Third Party"], label="Fault"),
        gr.Dropdown(['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision', 'Utility - Collision', 'Utility - Liability', 'Sport - All Perils'], label="Policy Type"),
        gr.Dropdown(["Sport", "Utility", "Sedan"], label="Vehicle Category"),
        gr.Dropdown(["less than 20,000", "20,000 to 29,000", "30,000 to 39,000", "40,000 to 59,000", "60,000 to 69,000", "more than 69,000"], label="Vehicle Price"),
        gr.Slider(1, 99999, step=1, label="Policy Number"),
        gr.Slider(1, 15, step=1, label="Rep Number"),
        gr.Slider(300, 700, step=100, label="Deductible"),
        gr.Slider(1, 4, step=1, label="Driver Rating"),
        gr.Dropdown(["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"], label="Days Policy Accident"),
        gr.Dropdown(["none", "8 to 15", "15 to 30", "more than 30"], label="Days Policy Claim"),
        gr.Dropdown(["none", "1", "2 to 4", "more than 4"], label="Past Number of Claims"),
        gr.Dropdown(["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"], label="Age of Vehicle"),
        gr.Dropdown(["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"], label="Age of Policy Holder"),
        gr.Radio(["No", "Yes"], label="Police Report Filed"),
        gr.Radio(["No", "Yes"], label="Witness Present"),
        gr.Radio(["External", "Internal"], label="Agent Type"),
        gr.Dropdown(["none", "1 to 2", "3 to 5", "more than 5"], label="Number of Suppliments"),
        gr.Dropdown(["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"], label="Address Change Claim"),
        gr.Dropdown(["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"], label="Number of Cars"),
        gr.Slider(1994, 1996, step=1, label="Year"),
        gr.Dropdown(["Liability", "Collision", "All Perils"], label="Base Policy")
    ],
    outputs=gr.Text(label="Prediction"),
    title="Insurance Fraud Detection",
    description="Fill in the claim details to get a fraud probability prediction."
)

# we'll build the interface next
if __name__ == "__main__":
    demo.launch(share=True)