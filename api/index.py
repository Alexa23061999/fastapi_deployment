from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel
from sklearn.datasets import load_iris


target_names = load_iris().target_names
app = FastAPI()


class Input(BaseModel):
    features: list


try:
    md = joblib.load('./api/mlmodel.joblib')
except Exception as e:
    print(f"Error loading with joblib: {e}")
    with open('./api/mlmodel.joblib', 'rb') as f:
        md = pickle.load(f)


@app.post('/predict')

def predict(data: Input):

    x_input = np.array(data.features).reshape(1, -1)
    prediction = md.predict(x_input)
    return {"prediction": target_names[prediction[0]]}
