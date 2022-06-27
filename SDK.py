#%% Import
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn  # API伺服器
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import math 
import os
import sys
import traceback
from datetime import datetime
app = FastAPI(
    title="文本分類API",
    version="v1.0.0 TingAI",
    description="""TransFormer->用於衛生局文本分類
  """,
)
#%% Initial
model = AutoModelForSequenceClassification.from_pretrained("ShihTing/HealthBureauSix")
tokenizer = AutoTokenizer.from_pretrained("ShihTing/HealthBureauSix")

from typing import List, Optional
from pydantic import BaseModel, Field
#輸入AISP規則-單一輸出
class Prediction(BaseModel):
    Cat: str = Field(None, Description="文本類別")
    Conf: float = Field(None, Description="模型信度")
    ConfShow: str = Field(None, Description="模型信度")
#輸入AISP規則-多筆輸出
class InferenceResponseModel(BaseModel):
    status: bool = Field(None, description="回傳狀態")
    data: Optional[List[Prediction]] = Field(None, description="預測結果列表")
#輸入AISP規則-輸入
class Article(BaseModel):
    text: str = Field(None, description="文本內容")
class InferenceRequestModel(BaseModel):
    articles: List[Article] = Field(None, description="待分類文本列表")

Classification=['其他','服務態度問題','費用爭議','醫療品質','醫療機構管理問題','醫療爭議']

#%% API
@app.get("/time")
async def get_time():
    current_time = str(datetime.now()).split(".")[0]
    return current_time

@app.post("/PredictCallOne/") #單一輸入，回傳僅最高類別
def get_predictionsCallOne(query: InferenceRequestModel):
    if(len(query.articles[0].text)>505):
        query.articles[0].text = query.articles[0].text[0:505]
    inputs = tokenizer(query.articles[0].text, return_tensors="pt")
    outputs = model(**inputs)
    Prob = outputs.logits.softmax(dim=-1).tolist()

    pred = Prediction(
        Cat=Classification[Prob[0].index(max(Prob[0]))],
        Conf=max(Prob[0]),
        ConfShow='*'*math.ceil(max(Prob[0])*100)
    )

    json_compatible_item_data = jsonable_encoder(pred)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/PredictAll")
def get_predictionsAll(query: InferenceRequestModel):
    predictions = []
    for Question in query.articles:
        if(len(Question.text)>505):
            Question.text = Question.text[0:505]
        inputs = tokenizer(Question.text, return_tensors="pt")
        outputs = model(**inputs)
        Prob = outputs.logits.softmax(dim=-1).tolist()
    
        pred = Prediction(
            Cat=Classification[Prob[0].index(max(Prob[0]))],
            Conf=max(Prob[0]),
            ConfShow='*'*math.ceil(max(Prob[0])*100)
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



