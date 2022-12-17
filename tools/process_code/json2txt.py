# -*- coding: utf-8 -*-
# @Time : 2022/7/5 17:16
# @Author : qingquan
# @File : transformer_data.py
import json
import argparse
import re

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input DeepSpeed Checkpoint folder')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def transformer_data(file_path=parse_arguments().input_folder):
    with open(file_path, 'r') as rf, open(file_path.replace('json', 'txt'), 'w') as wf:
        contents = json.load(rf)
        for item in contents:
            item = [x.replace('\\','') for x in item]
            wf.write(json.dumps(item, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    transformer_data()
