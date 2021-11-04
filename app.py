from fastapi import FastAPI, File, Form, UploadFile, Query
import shutil
from fastapi.param_functions import Query
import pandas as pd
from pandas.core.reshape.melt import lreshape
from pandas_profiling import ProfileReport
from fastapi.responses import FileResponse
from datetime import datetime
import pickle
from typing import List

import numpy as np

app = FastAPI()

@app.post("/UploadCSV/")
async def Upload_CSV(file: UploadFile = File(...)):
    with open("dataset.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv("dataset.csv")
    colunas = str(df.columns.values.tolist())

    return f"O dataset foi carregado e possui as colunas {colunas}"


@app.post("/Analisador/")
async def Recebe_CSV_Gera_Relatorio(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(file.filename)
    analise = ProfileReport(df)
    data = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    
    analise.to_file(f"analise_{data}.html")
    FileResponse(f"analise_{data}.html")

    return f"O Relatório analise_{data}.html foi salvo"

@app.post("/TreinaClassificador-GaussianNB/")
async def Treina_Classificador(target: str = Form(...)):

    from sklearn.naive_bayes import GaussianNB

    df = pd.read_csv("dataset.csv")
    
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]

    GNB = GaussianNB()
    GNB.fit(X,y.values.ravel())
    score = str(round(GNB.score(X,y)*100,2))+"%"

    pkl_filename = "GNB_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(GNB, file)
        
    atributos = str(X.columns.values.tolist())
    return f"O modelo foi treinado com atributos: {str(atributos)}, target: {target} e {score} de acurácia média"

@app.post("/TreinaClassificador-LogisticRegression/")
async def Treina_Classificador(target: str = Form(...)):

    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv("dataset.csv")
    
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]

    LR = LogisticRegression()
    LR.fit(X,y.values.ravel())
    score = str(round(LR.score(X,y)*100,2))+"%"

    pkl_filename = "LR_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(LR, file)
        
    atributos = str(X.columns.values.tolist())
    return f"O modelo foi treinado com atributos: {str(atributos)}, target: {target} e {score} de acurácia média"

@app.post('/InferenciaGNB/')
async def predict(q: list = Query([])):
    lista = []
    for i in q:
        lista.append(i)

    atributos = [lista]
    print(atributos)

    # atributos = pd.DataFrame(lista)

    pkl_filename = "GNB_model.pkl"
    with open(pkl_filename, 'rb') as file:
        GNB = pickle.load(file)

    # #pred1 = GNB.predict(atributos)[0]
    pred2 = GNB.predict(np.array([lista],dtype=float))

    return pred2,pred2

@app.post('/InferenciaLR/')
async def predict(atributos: list = Query([])):
    lista = []
    for i in atributos:
        lista.append(i)

    atributos = pd.DataFrame(lista)

    pkl_filename = "LR_model.pkl"
    with open(pkl_filename, 'rb') as file:
        LR = pickle.load(file)

    pred = LR.predict(atributos)[0]

    return pred

