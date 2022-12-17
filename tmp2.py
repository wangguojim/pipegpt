# -*- coding: utf-8 -*-
# @Time : 2022/7/23 13:41
# @Author : qingquan
# @File : tmp2.py
import json
import os
import re
from tqdm import tqdm
# pat = re.compile(r'')

# import os
#
# input_path = '/raid/nlp/data/meta_data_1.7b/lccc_large/txt_files/'
# texts = os.listdir(input_path)
# for name in texts:
#     with open(input_path + name, 'r+', encoding='utf-8') as f1:
#         content = f1.read()
#         content = content.replace('content', 'text')
#         f1.seek(0, 0)
#         f1.write(content)
# print("done")

data_path = '/raid/nlp/data/meta_data_1.7b/lccc_large/uncompressed_files'
all_file_names = [os.path.join(root, f) for root, dirs, files in os.walk(data_path) for f in files]  # 递归遍历目录下文件

# print(all_file_names)
for file in tqdm(all_file_names):
    w_file = file.replace('uncompressed_files', 'txt_files')
    if not os.path.exists(os.path.split(w_file)[0]):
        os.makedirs(os.path.split(w_file)[0])
    with open(file, 'r') as rf, open(w_file, 'w') as wf:
        dialogues = json.load(rf)
        for dia in dialogues:
            this_dia = ''
            for i, sen in enumerate(dia):
                if i % 2:
                    this_dia += '[SPE_2]'
                else:
                    this_dia += '[SPE_1]'
                this_dia += sen.replace(' ', '')
            # print(this_dia)
            wf.write(json.dumps({'text': this_dia}, ensure_ascii=False) + '\n')