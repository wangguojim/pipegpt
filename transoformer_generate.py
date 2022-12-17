# -*- coding: utf-8 -*-
# @Time : 2022/6/30 18:04
# @Author : qingquan
# @File : transoformer_generate.py

# from transformers import GPT2ForSequenceClassification
# from transformers import BertTokenizer, GPT2Model
# tokenizer = BertTokenizer.from_pretrained('ckpts/transformers_25000')
# model = GPT2Model.from_pretrained('ckpts/transformers_25000/')
# exit()
#
# from transformers import pipeline, set_seed, GPT2Model
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# # generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# res = generator("你好啊", max_length=100, num_return_sequences=1)
# print(res)

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
