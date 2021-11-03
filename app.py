from fastapi import FastAPI, File, UploadFile
import shutil
import pandas as pd
from pandas_profiling import ProfileReport
from fastapi.responses import FileResponse

from datetime import datetime

app = FastAPI()

@app.post("/Analisador/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(file.filename)
    analise = ProfileReport(df)
    data = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    
    analise.to_file(f"analise_{data}.html")
    FileResponse(f"analise_{data}.html")

    return "Análise Executada - tratatrtrartatratratra"



#     from fastapi import FastAPI, File, Form, UploadFile

# app = FastAPI()


# @app.post("/files/")
# async def create_file(
#     file: bytes = File(...), fileb: UploadFile = File(...), token: str = Form(...)
# ):
#     return {
#         "file_size": len(file),
#         "token": token,
#         "fileb_content_type": fileb.content_type,
#     }