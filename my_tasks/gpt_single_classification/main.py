# -*- coding: utf-8 -*-
# @Time : 2022/5/9 21:36
# @Author : qingquan
# @File : main.py
from typing import Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from predict import Predictor
from log import Logger

logger = Logger(save_mode='rotat', log_path='logs/').logger
predictor = Predictor('fine_tuned_model_path', device='cpu')
app = FastAPI()


class Item(BaseModel):
    server: str
    client: str


@app.post("/digital_man/")
async def predict(item: Item):
    logger.info(f'input:    {dict(item)}')
    res, confidence = predictor.predict([item.server, item.client])
    return_dict = {'res': res, 'confidence': confidence}
    logger.info(f'output:   {return_dict}')
    return return_dict


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)