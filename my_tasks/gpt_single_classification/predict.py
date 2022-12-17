# -*- coding: utf-8 -*-
# @Time : 2022/5/9 20:28
# @Author : qingquan
# @File : predict.py
# @Software: PyCharm
import os
import torch
import pandas as pd
import torch.nn as nn
from transformers import BartForSequenceClassification, BertTokenizer
from config import get_config


class Predictor:
    def __init__(self, model_path, label_file='data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx', device='cpu'):
        self.model = BartForSequenceClassification.from_pretrained(
                    model_path, num_labels=9,  problem_type='single_label_classification').to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = device
        keyword2content = pd.read_excel(label_file, sheet_name='回答类别及内容-纠正')
        keyword = keyword2content[keyword2content.columns[0]].tolist()
        self.id2class = {k: v for k, v in enumerate(keyword)}

    def process(self, text):
        # text = (server, client)
        text = '[SEP]'.join(text).strip()
        input_ids = self.tokenizer.encode(text)
        return input_ids

    def predict(self, text):
        input_ids = self.process(text)
        input_ids = torch.tensor([input_ids]).to(self.device)
        logits = self.model(input_ids=input_ids).logits
        logits = torch.softmax(logits, dim=1)
        confidence, indices = logits.max(dim=1)
        return self.id2class[indices.item()], confidence.item()


if __name__ == '__main__':
    predictor = Predictor('fine_tuned_model_path', device='cpu')
    sens = [('好的，请问您还有其他业务需要办理吗？您可以跟我说查套餐.查语音.查流量等\u3000[SEP]我不要', '否定'),
     ('好的，正在为您办理XXX业务，业务套餐为XXX元xxx分钟包含XX兆流量，请稍候[SEP]没听见', '默认'), ('您的XXX业务办理成功，请问您还有其他业务需要办理查询吗？\u3000[SEP]嗯', '肯定'),
     ('XX先生/女生，您好，欢迎光临中国移动，是否帮您查询名下移动号卡业务，[SEP]没什么问题了', '肯定'),
     ('您的语音套餐为XXX分钟、已使用XXX分钟、剩余XXX分钟；为了让您有更好的使用体验，推荐您使用XXX语音加油包，加油包包含xxx元xxx分钟，您需要办理吗？[SEP]暂时没打算', '不办理'),
     ('已为您查询到您名下宽带套餐xxx，宽带电视数量XXX即将到期，[SEP]首页', '返回主屏幕'), ('业务查询办理成功，请问您还有其他业务需要办理么?[SEP]好的谢谢', '肯定'),
     ('好的，正在为您办理XXX宽带套餐业务，请稍等。[SEP]你刚刚说什么了', '默认'), ('业务查询办理成功，请问您还有其他业务需要办理么?[SEP]再说一下', '默认'),
     ('为您推荐最优惠的宽带套餐，以及丰富的智能家居产品，是否需要向您介绍下我们的优惠套餐以及智能家居产品呢？[SEP]不打算做', '否定')]
    for sen in sens:
        text = sen[0].split('[SEP]')
        print(predictor.predict(text), sen[1])


