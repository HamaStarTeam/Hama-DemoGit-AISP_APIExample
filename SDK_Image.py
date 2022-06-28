#%% Import
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn  # API伺服器
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import Response
import math 
import os
import sys
from datetime import datetime
app = FastAPI(
    title="Yolo範例",
    version="v1.0.0 TingAI",
    description="""影像包裝成AISP的範例
  """,
)
import torch
import numpy as np
import io
from imageio import v3 as iio
from PIL import Image
#%% Initial
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

from typing import List, Optional
from pydantic import BaseModel, Field
#輸入AISP規則-單一輸出
class Prediction(BaseModel):
    Cat: str = Field(None, Description="偵測結果")
#輸入AISP規則-多筆輸出
class InferenceResponseModel(BaseModel):
    status: bool = Field(None, description="回傳狀態")
    data: Optional[List[Prediction]] = Field(None, description="預測結果列表")
#輸入AISP規則-輸入
class Article(BaseModel):
    ImageUrl: str = Field(None, description="影像來源")
    IsImageOut : bool = Field(None, description="是否回傳影像")
class InferenceRequestModel(BaseModel):
    articles: List[Article] = Field(None, description="影像來源列表")

#%% API
@app.get("/time")
async def get_time():
    current_time = str(datetime.now()).split(".")[0]
    return current_time

@app.post("/PredictCallOne/") #單一輸入，回傳僅最高類別
def get_predictionsCallOne(query: InferenceRequestModel):
    img = query.articles[0].ImageUrl
    results = model(img)

    pred = Prediction(
        Cat=str(results.pandas().xyxy[0]),
    )
    
    if query.articles[0].IsImageOut:
        # 影像輸出 results.render() get image list
        im = np.squeeze(results.render())
        # img = Image.fromarray(im).resize((512, 512))
        new_width  = 512
        new_height = int(new_width * im.shape[0] / im.shape[1] )
        img = Image.fromarray(im).resize((new_width, new_height))
        with io.BytesIO() as buf:
            # pillow to BytesIo
            iio.imwrite(buf, img, plugin="pillow", format="JPEG")
            im_bytes = buf.getvalue()
        return Response(im_bytes, media_type='image/jpeg')
    else:
        #Json輸出->任務用
        json_compatible_item_data = jsonable_encoder(pred)
        return JSONResponse(content=json_compatible_item_data)
    
    

@app.post("/PredictAll")
def get_predictionsAll(query: InferenceRequestModel):
    predictions = []
    for Question in query.articles:
        img = Question.ImageUrl
        results = model(img)
        pred = Prediction(
            Cat=str(results.pandas().xyxy[0]),
        )
        predictions.append(pred)
        
    response = InferenceResponseModel(
        status=True,
        data=predictions,
    )
    
    return response
#%% Run main
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5007) #部署時使用



