from typing import Union, Optional, List

from fastapi import FastAPI, Request, File, UploadFile, Form, Response, status, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import pandas as pd
from sklearn import preprocessing
from io import BytesIO

import pickle
import settings

app = FastAPI()

# Load the model from disk
knn_model = pickle.load(open(settings.models_dir + '/knn_10_folds', 'rb'))
rf_model = pickle.load(open(settings.models_dir + '/rf_10_folds', 'rb'))
dt_model = pickle.load(open(settings.models_dir + '/dt_10_folds', 'rb'))

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/dummypath")
async def get_body(request: Request):
    input = await request.json()
    print('RECEIVED MESSAGE:', input)
    return {'status': 'hello', 'message': input}


@app.post("/predictions/driver_identification")
async def driver_identification(request: Request):
    scaler = preprocessing.MinMaxScaler()
    # Getting min-max data from database to normalize incoming data
    '''Predicting by sent request
    '''
    data = await request.json()
    df = pd.read_json(data)

    result_knn = knn_model.predict(df)
    result_rf = rf_model.predict(df)
    result_dt = dt_model.predict(df)
    print(result_knn, result_dt, result_rf)
    return data
    # Normalize incoming stream data
    # df_norm = df(scaler.fit_transform(payload), columns=payload.columns)
    
    # result = loaded_model.predict(df_norm.iloc[0][0])
    # if result >= 0.75:
    #     return {"result": True, "message": "Driver is in database", "score": result}

    # return {"result": False, "message": "Driver is NOT in database", "score": result}


@app.post("/predictions/upload_csv")
def upload(file: UploadFile = File(...)):
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    file.file.close()
    result_knn = knn_model.predict(df)
    result_rf = rf_model.predict(df)
    result_dt = dt_model.predict(df)
    # df.to_dict(orient='records')
    return {'results': {'knn': result_knn.tolist(), 'rf': result_rf.tolist(), 'dt': result_dt.tolist()}}