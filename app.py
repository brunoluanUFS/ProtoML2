from fastapi import FastAPI, File, Form, UploadFile, Query
import shutil
import pandas as pd
from pandas_profiling import ProfileReport
from fastapi.responses import FileResponse
from datetime import datetime
import pickle
import numpy as np

app = FastAPI()

@app.post("/UploadCSV/")
async def Upload_CSV(file: UploadFile = File(...)):
    with open("dataset.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv("dataset.csv")
    colunas = str(df.columns.values.tolist())

    return f"O dataset foi carregado e possui {len(df.columns.values.tolist())} colunas {colunas}"


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
    return f"O modelo foi treinado e salvo no servidor com atributos: {str(atributos)}, target: {target} e {score} de acurácia média"

@app.post("/TreinaClassificador-LogisticRegression/")
async def Treina_Classificador(target: str = Form(...)):

    from sklearn.linear_model import LogisticRegression
    df = pd.read_csv("dataset.csv")
    # PREPROCESSAMENTO

    # df_numeric = df._get_numeric_data()
    # print(df_numeric)
    # cols = df.columns
    # num_cols = df._get_numeric_data().columns
    # cat_cols = list(set(cols) - set(num_cols))

    # ONEHOT
    

    # TREINA O MODELO
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]
    LR = LogisticRegression()
    LR.fit(X,y.values.ravel())
    score = str(round(LR.score(X,y)*100,2))+"%"
    pkl_filename = "LR_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(LR, file)
    atributos = str(X.columns.values.tolist())
    return f"O modelo foi treinado e salvo no servidor com atributos: {str(atributos)}, target: {target} e {score} de acurácia média"

@app.post('/InferenciaGNB/')
async def predictGNB(q: list = Query([])):
    q2 = []
    for i in q:
        q2.append(np.float_(i))

    pkl_filename = "GNB_model.pkl"
    with open(pkl_filename, 'rb') as file:
        GNB = pickle.load(file)

    pred = GNB.predict([q2])
    return str(pred)

@app.post('/InferenciaLR/')
async def predictLR(q: list = Query([])):
    q2 = []
    for i in q:
        q2.append(np.float_(i))

    pkl_filename = "LR_model.pkl"
    with open(pkl_filename, 'rb') as file:
        LR = pickle.load(file)
    
    print(q,[q2],len(q),len(q2))

    pred = LR.predict([q2])
    return str(pred)


### PARTE DO HTML