from fastapi import FastAPI, File, Form, UploadFile, Query
import shutil
from fastapi.param_functions import Query
import pandas as pd
from pandas_profiling import ProfileReport
from fastapi.responses import FileResponse
from datetime import datetime
import pickle

from typing import List

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

@app.post("/TreinaClassificador/")
async def Treina_Classificador(target: str = Form(...)):

    from sklearn import svm
    from sklearn.naive_bayes import GaussianNB

    df = pd.read_csv("dataset.csv")
    
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]

    clf = GaussianNB()
    clf.fit(X,y.values.ravel())
    score = str(round(clf.score(X,y)*100,2))+"%"

    pkl_filename = "clf_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
        
    atributos = str(X.columns.values.tolist())
    return f"O modelo foi treinado com atributos: {str(atributos)}, target: {target} e {score} de acurácia média"

@app.post('/InferenciaClassificador/')
async def predict(atributos: list = Query([])):
    lista = []
    for i in atributos:
        lista.append(i)

    atributos = pd.DataFrame(lista)

    pkl_filename = "clf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    pred = clf.predict(atributos)[0]

    return pred