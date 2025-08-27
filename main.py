# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Load the trained model
model = joblib.load("model.joblib")

# 2. Define the structure of the JSON data you expect to receive
class ApplicantData(BaseModel):
    person_income: float
    person_age: int
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float

# 3. Create the FastAPI app
app = FastAPI(title="Credit Scoring API", version="0.1")

# 4. Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Scoring API"}

# 5. Define the prediction endpoint
@app.post("/predict")
def predict(data: ApplicantData):
    # Convert the received JSON into a pandas DataFrame in the exact same format as training
    input_data = pd.DataFrame([dict(data)])

    # Get the model's prediction (0 or 1)
    prediction = model.predict(input_data)[0]

    # Get the probability of the prediction
    probability = model.predict_proba(input_data)[0].tolist()

    # Return the result as JSON
    return {
        "prediction": int(prediction), # 0 or 1
        "probability_default": probability[1], # Prob that it's 1 (bad)
        "probability_repay": probability[0]  # Prob that it's 0 (good)
    }