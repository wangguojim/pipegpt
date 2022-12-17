# -*- coding: utf-8 -*-
# @Time : 2022/7/20 15:36
# @Author : qingquan
# @File : get_data_path.py
import os
import warnings
warnings.filterwarnings("ignore")
from megatron.data.gpt_dataset import get_indexed_dataset_


path = '/raid/nlp/data/aggregated_meta_data'
all_file_paths = [os.path.join(root, f) for root, dirs, files in os.walk(path) for f in files]


# 每次使用前缀都删除之前运行生成的文件
for file_name in all_file_paths:
    if 'train' in file_name or 'valid' in file_name or 'test' in file_name:
        os.remove(file_name)


dedup_all_file_paths = list(set([i[:-4] for i in all_file_paths if not i.endswith('txt')]))
dedup_all_file_paths = [i for i in dedup_all_file_paths if 'train' not in i or 'valid' not in i or 'test' not in i]#[:50]
# print(dedup_all_file_paths)
# print(len(dedup_all_file_paths))

str = ''
total_samples = 0
for i in dedup_all_file_paths:
    weight = get_indexed_dataset_(i, 'infer', True).sizes.shape[0]
    str += f'{weight} {i} '
    total_samples += weight
print(str)
print(total_samples)  # 89620624
