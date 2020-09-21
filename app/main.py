from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
import joblib

MODEL = '../models/classifier.pkl'

app = FastAPI()

classifier = None

@app.on_event("startup")
def load_classifier():
    global classifier
    classifier = joblib.load(MODEL)

@app.get("/classify/")
def classify(numeric0: Optional[int] = Query(None,
                                             alias="numeric0",
                                             title="Numeric 0",
                                             description="Numeric value. Can be any integer"),
             categorical0: Optional[str] = Query('c',
                                                 alias="categorical0",
                                                 title="Categorical 0",
                                                 description="Categorical0 value. Can be 'a', 'b', or 'c'",
                                                 max_length=1,
                                                 regex="^[a-c]$")):
    x = pd.DataFrame({"numeric0": [numeric0], "categorical0": [categorical0]})
    result = classifier.predict(x)
    return {"result": str(result[0])}