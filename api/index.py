from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel
from sklearn.datasets import load_iris


target_names = load_iris().target_names
app = FastAPI()


class Input(BaseModel):
    features: list


md = joblib.load('./api/mlmodel.joblib')

@app.post('/predict')

def predict(data: Input):

    x_input = np.array(data.features).reshape(1, -1)
    prediction = md.predict(x_input)
    return {"prediction": target_names[prediction[0]]}