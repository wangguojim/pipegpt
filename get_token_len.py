# -*- coding: utf-8 -*-
# @Time : 2022/7/31 13:18
# @Author : qingquan
# @File : get_token_len.py
import os
import numpy as np
from megatron.data.indexed_dataset import make_dataset

processed_data_dir = '/raid/nlp/data/aggregated_meta_data_1.7b'
idx_files = [os.path.join(root, f) for root, dirs, files in os.walk(processed_data_dir) for f in files if f.endswith('.idx')]
prefix_files = [file.replace('.idx', '') for file in idx_files]
print(len(idx_files))


total_token_len = 0
for idx_file in prefix_files:
    indexed_dataset = make_dataset(idx_file, 'mmap', skip_warmup=False)
    sizes = indexed_dataset.sizes
    token_len = np.sum(sizes)
    print(token_len)
    total_token_len += token_len
print(total_token_len)
