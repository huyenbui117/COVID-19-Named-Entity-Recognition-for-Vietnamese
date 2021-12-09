import fastapi
import pandas as pd
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import *

app = fastapi.FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(path="/demo", response_class=FileResponse)
async def main(file: UploadFile = File(media_type='multipart', default='Any')):
    file_location = "sentences.txt"
    with open(file_location, "wb") as file_object:
        file_object.write(file.file.read())
    run()
    return "predict_sentences.json"
