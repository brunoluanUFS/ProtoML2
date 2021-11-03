from fastapi import FastAPI, File, Form, UploadFile
import shutil
import pandas as pd
from pandas_profiling import ProfileReport
from fastapi.responses import FileResponse
from datetime import datetime

app = FastAPI()

@app.post("/Relatorio/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(file.filename)
    analise = ProfileReport(df)
    data = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    
    analise.to_file(f"analise_{data}.html")
    FileResponse(f"analise_{data}.html")

    return f"O Relatório analise_{data}.html foi salvo"

@app.post("/Classificador/")
async def dataset():

    df2 = pd.read_csv("dataset.csv")
    txt2 = str(df2.columns)

    return txt2

    # return f"{target}"

# from fastapi import FastAPI
# import uvicorn
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# from pydantic import BaseModel

# # Creating FastAPI instance
# app = FastAPI()

# # Creating class to define the request body
# # and the type hints of each attribute
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
# 	return { 'class' : iris.target_names[class_idx]}
