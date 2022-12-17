# -*- coding: utf-8 -*-
# @Time : 2022/3/21 18:37
# @Author : qingquan
# @File : config.py
from argparse import ArgumentParser


# 超参配置
def get_config():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--max_len', type=int, default=512, help='max len of the sentence')
    parser.add_argument('--batch_size', type=int, default=32, help='dataloader batch size')
    parser.add_argument('--split_rate', type=float, default=0.8)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_path', type=str, required=True, default=None)
    parser.add_argument('--device', type=str, required=True, default=None)
    parser.add_argument('--base_params', type=str, required=True, default=None)
    parser.add_argument('--special_token_file', type=str, required=True, default=None)
    parser.add_argument('--save_path', type=str, required=True, default=None)
    parser.add_argument('--tensorboard_path', type=str, required=True, default=None)

    args = parser.parse_args()
#     args = parser.parse_args(args=[])
    return args