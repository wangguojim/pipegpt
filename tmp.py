# -*- coding: utf-8 -*-
# @Time : 2022/6/27 22:36
# @Author : qingquan
# @File : tmp.py
import torch
from collections import OrderedDict
from transformers import BertTokenizer
# # BertTokenizer.convert_ids_to_tokens()
# state_dict = torch.load('ckpts/megatron_7000/iter_0007000/mp_rank_00/model_optim_rng.pt')
#
# state_dict['model']['language_model']['encoder'] = OrderedDict([(k.replace('self_attention', 'attention'),v) for k,v in state_dict['model']['language_model']['encoder'].items()])
# # for key,value in state_dict['model']['language_model']['encoder'].items():
# #     key = key.replace('self_attention', 'attention')
# #     state_dict['model']['language_model']['encoder'][key] = value
# print(state_dict['model']['language_model']['encoder'])
#
# torch.save(state_dict, 'ckpts/megatron_7000_modified/iter_0007000/mp_rank_00/model_optim_rng.pt')

# import time
# for i in range(100):
#     print("\r", i, end="", flush=True)
#     time.sleep(1)
# import os
# file_path = '/raid/nlp/data/meta_data/sohu/sohu-20091019-20130819.rar'
# print(os.path.getsize(file_path)/1024/1024/1024)
import json
import os
from tqdm import tqdm

data_path = f'/raid/nlp/data/meta_data'  # 原始数据集位置
all_file_names = [os.path.join(root, f) for root, dirs, files in os.walk(data_path) for f in files if f.endswith('.txt')
                  and 'txt_files' in os.path.join(root, f)] # 递归遍历目录下文件
error_files = []

# for file in tqdm(all_file_names):
#     with open(file) as rf:
#         try:
#             for i, json_data in enumerate(rf):
#                 try:
#                     json_data = json.loads(json_data)
#                     tmp = json_data['text']
#                 except Exception as e:
#                     error_files.append(file)
#                     print(i, e)
#                     print(file)
#                     print(json_data)
#         except Exception as e:
#             print(e)
#             error_files.append(file)
#             print(file)
# dedup_error_files = set(list(error_files))
# print(dedup_error_files)
# print(len(dedup_error_files))

for file in tqdm(all_file_names):
    with open(file) as rf:
            for i, json_data in enumerate(rf):
                    json_data = json.loads(json_data)
                    tmp = json_data['text']


