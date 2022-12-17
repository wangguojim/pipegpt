# -*- coding: utf-8 -*-
# @Time : 2022/4/22 14:03
# @Author : qingquan
# @File : digital_process.py
import math
import torch
import random
import numpy as np
import copy
import os
import yaml
from config import get_config
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, data_path, mode='train', seed=1234, split_ratio=0.8):
        random.seed(seed)
        self.args = get_config()
        keyword2content = pd.read_excel(data_path, sheet_name='回答类别及内容-纠正')
        keyword, content = keyword2content[keyword2content.columns[0]].tolist(),\
                           keyword2content[keyword2content.columns[1]].apply(self.split).tolist()
        self.class2id = {v: k for k, v in enumerate(keyword)}
        self.keyword2content = dict(zip(keyword, content))
        for k, v in self.keyword2content.items():
            random.shuffle(v)
        if mode=='train':
            self.keyword2content = {k:v[:int(len(v)*split_ratio)] for k, v in self.keyword2content.items()}
        elif mode=='test':
            self.keyword2content = {k:v[int(len(v)*split_ratio):] for k, v in self.keyword2content.items()}
        one_trun = pd.read_excel(data_path, sheet_name='业务和回答类别')[['客服', '类别']].values.tolist()
        self.data = []
        for speaker1, speaker2 in one_trun:
            for item in self.keyword2content[speaker2]:
                self.data.append((f'{speaker1}[SEP]{item}', speaker2))
        tmp = copy.deepcopy(self.data)
        random.shuffle(tmp)
        # print(tmp[:10])
        # 添加特殊token
        special_tokens = yaml.load(open(os.path.join(self.args.base_params, self.args.special_token_file)),
                                        Loader=yaml.FullLoader)['special_tokens']
        self.tokenizer = BertTokenizer.from_pretrained(self.args.base_params, additional_special_tokens=special_tokens)
        # self.tokenizer = BertTokenizer.from_pretrained('../../ch_tokenizer_data/vocab.txt',
        #                                                additional_special_tokens=['[NUM]', '[E_WORD]'])

    def split(self, string):
        import re
        pattern = re.compile(r'[,，。、]')

        def corr(string):
            res = re.split(pattern, string)
            res = [i.strip() for i in res if i != '']
            res = sorted(set(res), key=res.index)
            return res
        return corr(string)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        source, target = self.data[item]
        source_ids = self.tokenizer.encode(source)[1:]
        target_ids = self.class2id[target]
        return torch.tensor(source_ids), target_ids
#         return source, target


def paired_collate_fn(insts):
    enc_input = [src for src, tgt in insts]
    labels = [tgt for src, tgt in insts]
    # encoder 打padding
    enc_input_packed = pack_sequence(enc_input, enforce_sorted=False)
    enc_input_padded, word_lens = pad_packed_sequence(enc_input_packed, batch_first=True)  # 序列打padding
    enc_mask = enc_input_padded.gt(0)
    return enc_input_padded, enc_mask, labels


if __name__ == '__main__':
    import time
    from transformers import BartForSequenceClassification
    # model = BartForSequenceClassification.from_pretrained(
    #     'fine_tuned_model_path', num_labels=9, problem_type='single_label_classification')
    bart_data = MyDataset('data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx')
    # for i in bart_data:
    #     pass
    data_loader = DataLoader(bart_data, batch_size=4, collate_fn=paired_collate_fn)
    for i, item in enumerate(data_loader):
        print(item)
        break
        # enc_input_padded, enc_mask, labels = item
        # t1 = time.time()
        # print(src.shape)
        # labels = torch.Tensor(labels).long()
        # outputs = model(enc_input_padded, enc_mask, labels=labels)
        # loss, logits = outputs[:2]
        # print(loss)
        # print(time.time() - t1)