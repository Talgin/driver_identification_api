from typing import Union, Optional, List

from fastapi import FastAPI, Request, File, UploadFile, Form, Response, status, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import pandas as pd
import numpy as np
from statistics import mode
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


def most_common(List):
    return mode(List)

@app.post("/predictions/driver_identification")
async def driver_identification(request: Request):
    scaler = preprocessing.MinMaxScaler()
    # Getting min-max data from database to normalize incoming data
    '''Predicting by sent request
    '''
    data = await request.json()
    df = pd.read_json(data)

    # Prediction classes
    result_knn_cls = knn_model.predict(df)
    result_rf_cls = rf_model.predict(df)
    result_dt_cls = dt_model.predict(df)

    most_common_knn = most_common(result_knn_cls)
    most_common_rf = most_common(result_rf_cls)
    most_common_dt = most_common(result_dt_cls)
    # print(most_common_knn, most_common_rf, most_common_dt)
    # Prediction with probabilities
    result_knn = knn_model.predict_proba(df)
    result_rf = rf_model.predict_proba(df)
    result_dt = dt_model.predict_proba(df)
    # print(result_knn, result_dt, result_rf)

    # results_dct = dict()
    # for i in range(len(result_knn)):
    #     results_dct['knn_probabilities'] = [np.argmax(), np.max()]
    
    knn_flag = False
    rf_flag = False
    dt_flag = False
    # if result_knn >= settings.models_threshold:
    #     knn_flag = True
    # if result_rf >= settings.models_threshold:
    #     rf_flag = True
    # if result_dt >= settings.models_threshold:
    #     dt_flag = True

    # If all three predictions are above threshold and point to one ID return ID of driver
    # if (knn_flag and rf_flag and dt_flag) and (most_common_knn == most_common_rf == most_common_dt):
    if most_common_knn == most_common_rf == most_common_dt:
        # print('here')
        return {"result": True, "message": "Driver found", "values": int(most_common_rf)}
    else:
        # Else return each prediction probability and class
        return {"result": True, "message": "Driver probabilities", "values": [int(most_common_knn), int(most_common_rf), int(most_common_dt)]}
    
    # return data

    # Normalize incoming stream data
    # df_norm = df(scaler.fit_transform(payload), columns=payload.columns)
    
    # Get indices of highest probability
    # for i in range(len(result_knn)):
    #     np.argmax(result_knn[i]))

    # result = loaded_model.predict(df_norm.iloc[0][0])
    
    # If all three predictions are above threshold and point to one ID return ID of driver
    # if (knn_flag and rf_flag and dt_flag) and :

    # # Else return each prediction probability and class
    #     return {"result": True, "message": "Driver is in database", "score": result}

    # return {"result": False, "message": "Driver is NOT in database", "score": result}


@app.post("/predictions/upload_csv")
def upload(file: UploadFile = File(...)):
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    file.file.close()
    result_knn_cls = knn_model.predict(df)
    result_rf_cls = rf_model.predict(df)
    result_dt_cls = dt_model.predict(df)

    most_common_knn = most_common(result_knn_cls)
    most_common_rf = most_common(result_rf_cls)
    most_common_dt = most_common(result_dt_cls)
    # df.to_dict(orient='records')
    # return {'results': {'knn': result_knn.tolist(), 'rf': result_rf.tolist(), 'dt': result_dt.tolist()}}
    # If all three predictions are above threshold and point to one ID return ID of driver
    # if (knn_flag and rf_flag and dt_flag) and (most_common_knn == most_common_rf == most_common_dt):
    print(most_common_knn, most_common_rf, most_common_dt)
    if most_common_knn == most_common_rf == most_common_dt:
        # print('here')
        return {"result": True, "message": "Driver found", "values": int(most_common_rf)}
    else:
        # Else return each prediction probability and class
        return {"result": True, "message": "Driver probabilities", "values": [int(most_common_knn), int(most_common_rf), int(most_common_dt)]}