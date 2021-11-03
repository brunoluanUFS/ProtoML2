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

    from sklearn.naive_bayes import GaussianNB

    df = pd.read_csv("dataset.csv")
    
    X = df.loc[:, df.columns != target]
    Y = df.loc[:, df.columns == target]

    atributos = str(X.columns.values.tolist())

    clf = GaussianNB()
    clf.fit(X,Y)    # Saving the model to a serialized .pkl file
    

    pkl_filename = "../clf_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    return f"O modelo foi treinado com GaussianNB, com atributos {str(atributos)} e target {target}"

@app.post('/InferenciaClassificador/')
async def predict(atributos: list = Query([])):
    lista = []
    for i in atributos:
        lista.append(i)

    atributos = pd.DataFrame(lista)

    pkl_filename = "../clf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)

    pred = clf.predict(atributos)[0]
    prob = clf.pre

    return pred

# from pydantic import BaseModel

# # Creating class to define the request body and the type hints of each attribute
# class request_body(BaseModel):
# 	sepal_length : float
# 	sepal_width : float
# 	petal_length : float
# 	petal_width : float

# # Loading Iris Dataset
# iris = load_iris()

# # Getting our Features and Targets
# X = iris.data
# Y = iris.target

# # Creating and Fitting our Model
# clf = GaussianNB()
# clf.fit(X,Y)

# # Creating an Endpoint to receive the data
# # to make prediction on.
# @app.post('/predict')
# def predict(data : request_body):
# 	# Making the data in a form suitable for prediction
# 	test_data = [[
# 			data.sepal_length,
# 			data.sepal_width,
# 			data.petal_length,
# 			data.petal_width
# 	]]
	
# 	# Predicting the Class
# 	class_idx = clf.predict(test_data)[0]
	
# 	# Return the Result
#	return { 'class' : iris.target_names[class_idx]}