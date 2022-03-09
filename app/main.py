import json
import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel
import os


class Data(BaseModel):
    data: object


app = FastAPI()


@app.post("/healthcheck")
def hello(data: Data):
    """ Main page of the app. """
    return data


@app.get("/predict")
async def predict(url: str):
    """ Return JSON serializable output from the model """
    payload = {"url": url}
    return payload
