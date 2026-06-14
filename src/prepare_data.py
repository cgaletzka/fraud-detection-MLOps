import numpy as np
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    
    # drop variables
    drop_vars = ["PolicyNumber",
                 "AgeOfPolicyHolder",
                 "Deductible",
                 "Days:Policy-Accident",
                 "Days:Policy-Claim",
                 "WitnessPresent",
                 "BasePolicy",
                 "VehicleCategory",
                 "RepNumber"] 
    df = df.drop(drop_vars, axis=1)

    # group labels together
    ## Make
    group_labels = ('Accura',
                    'Ford',
                    'VW',
                    'Dodge',
                    'Saab',
                    'Mercury',
                    'Saturn',
                    'Nisson',
                    'BMW',
                    'Jaguar',
                    'Porche',
                    'Mecedes',
                    'Ferrari',
                    'Lexus') 
    df["Make"] = df["Make"].replace(group_labels, 'Other')

    ## MaritalStatus
    group_labels = ("Divorced","Widow")
    df["MaritalStatus"] = df["MaritalStatus"].replace(group_labels, "Other")

    ## Age
    df['Age'] = df['Age'].replace(0,16)

    ## PolicyType
    group_labels = ("Utility - Collision",
                    "Sport - All Perils",
                    "Utility - Liability",
                    "Sport - Liability")
    df["PolicyType"] = df["PolicyType"].replace(group_labels, "Other")

    ## VehiclePrice
    group_labels = ("more than 69,000",
                    "60,000 to 69,000")
    df['VehiclePrice'] = df['VehiclePrice'].replace(group_labels, "more than 60,000")

    ## AgeOfVehicle
    group_labels = ("2 years",
                    "3 years")
    df["AgeOfVehicle"] = df["AgeOfVehicle"].replace(group_labels, "between 2 and 3 years")

    ## AddressChange-Claim
    group_labels = ("under 6 months",
                    "1 year")
    df["AddressChange-Claim"] = df["AddressChange-Claim"].replace(group_labels, "6 months to 1 year")

    ## NumberOfCars
    group_labels = ("3 to 4",
                    "5 to 8",
                    "more than 8")
    df["NumberOfCars"] = df["NumberOfCars"].replace(group_labels, "more than 2 vehicles")

    # replace 0s with NaNs
    cols = ["DayOfWeekClaimed", "MonthClaimed"]
    df[cols] = df[cols].replace("0", np.nan)

    # change Year to categorical
    df['Year'] = df['Year'].astype(str)

    # encode binary variables
    df["AccidentArea"] = df["AccidentArea"].map({"Rural": 0, "Urban": 1})
    df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
    df["Fault"] = df["Fault"].map({"Third Party": 0, "Policy Holder": 1})
    df["PoliceReportFiled"] = df["PoliceReportFiled"].map({"No": 0, "Yes": 1})
    df["AgentType"] = df["AgentType"].map({"External": 0, "Internal": 1})
    df["FraudFound"] = df["FraudFound"].map({"No": 0, "Yes": 1})

    # encode ordinal variables
    mapping = {"none": 1,
               "1 to 2": 2,
               "3 to 5": 3,
               "more than 5": 4}
    
    df["NumberOfSuppliments"] = df["NumberOfSuppliments"].map(mapping)

    mapping = {"less than 20,000": 1,
               "20,000 to 29,000": 2,
               "30,000 to 39,000": 3,
               "40,000 to 59,000": 4,
               "more than 60,000": 5}
    
    df["VehiclePrice"] = df["VehiclePrice"].map(mapping)

    mapping = {"new": 1,
               "between 2 and 3 years": 2,
               "5 years": 3,
               "6 years": 4,
               "7 years": 5,
               "more than 7": 6}
    
    df["AgeOfVehicle"] = df["AgeOfVehicle"].map(mapping)

    mapping = {"no change": 1,
               "6 months to 1 year": 2,
               "2 to 3 years": 3,
               "4 to 8 years": 4}
    
    df["AddressChange-Claim"] = df["AddressChange-Claim"].map(mapping)

    mapping = {"1 vehicle": 1,
               "2 vehicles": 2,
               "more than 2 vehicles": 3}
    
    df["NumberOfCars"] = df["NumberOfCars"].map(mapping)

    mapping = {"1994": 1,
               "1995": 2,
               "1996": 3}
    
    df["Year"] = df["Year"].map(mapping)

    mapping = {"none": 1,
               "1": 2,
               "2 to 4": 3,
               "more than 4": 4}
    
    df["PastNumberOfClaims"] = df["PastNumberOfClaims"].map(mapping)

    # drop rows with missing values
    df = df.dropna()

    # change back to int
    df["AgeOfVehicle"] = df["AgeOfVehicle"].astype(int)

    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data("data/raw/carclaims.csv")
    df = clean_data(df)
    save_data(df, "data/processed/carclaims_processed.csv")