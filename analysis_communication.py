# -*- coding: utf-8 -*-
# @Time : 2022/11/11 14:11
# @Author : qingquan
# @File : analysis_communication.py
from tqdm import tqdm


kaiwu_path = 'kaiwu_gpt-1.3B-lr-1.2e-4-minlr-1.0e-6-bs-960-gpus-16-mp-1-pp-1-ep-16-mlc-0.01-cap-1.0-drop-false_dgx01gpunode_2022.11.11-15.24.30.log'
jiutian_path = 'jiutian_gpt-1.3B-lr-1.2e-4-minlr-1.0e-6-bs-960-gpus-8-mp-1-pp-1-ep-16-mlc-0.01-cap-1.0-drop-false_dl-1009251248-pod-jupyter-549488cfdd-tbd9k_2022.11.14-10.52.18.log'
jiutian_path_16_mp = 'jiutian_gpt-1.3B-lr-1.2e-4-minlr-1.0e-6-bs-960-gpus-16-mp-1-pp-1-ep-16-mlc-0.01-cap-1.0-drop-false_dl-1009150663-pod-jupyter-544f858644-xqnxg_2022.11.14-10.30.43.log'
jiutian_path_24_mp = 'jiutian_gpt-1.3B-lr-1.2e-4-minlr-1.0e-6-bs-960-gpus-24-mp-1-pp-1-ep-24-mlc-0.01-cap-1.0-drop-false_dl-1009150663-pod-jupyter-544f858644-xqnxg_2022.11.16-22.27.23.log'
use_path = f'all2all_test/{jiutian_path_24_mp}'
with open(use_path) as r_f:
    data = r_f.readlines()
    # use_lines = data[2306165:]
    # use_lines = data[1159616:]
    use_lines = data[780229:]
    # 自301次迭代开始耗时总量
    total_time = 0

    # all_to_all_single通信
    total_all2all_single_time = 0
    total_all2all_msg_size = 0
    # all_rudece通信
    total_allreduce_time = 0
    total_allreduce_msg_size = 0
    for line in tqdm(use_lines):
        if 'consumed tokens' in line:
            this_iter_time = float(line.split('|')[3].split(':')[1][1:-1])
            total_time += this_iter_time
        if 'comm op: all_to_all_single' in line:
            this_all2all_cost_time = float(line.split('|')[2].split(':')[1][1:-1])
            total_all2all_single_time += this_all2all_cost_time
            this_all2all_msg_size = float(line.split('|')[3].split(':')[1][1:-3])
            total_all2all_msg_size += this_all2all_msg_size
        if 'comm op: all_reduce' in line:
            this_allreduce_cost_time = float(line.split('|')[2].split(':')[1][1:-1])
            total_allreduce_time += this_allreduce_cost_time
            this_allreduce_msg_size = float(line.split('|')[3].split(':')[1][1:-3])
            total_allreduce_msg_size = this_allreduce_msg_size
print(f'total_time:{total_time}')
print(f'total_all2all_time: {total_all2all_single_time}')
print(f'total_all2all_msg_size: {total_all2all_msg_size}')
print(f'total_allreduce_time: {total_allreduce_time}')
print(f'total_allreduce_msg_size: {total_allreduce_msg_size}')
print(f'all2all_time占比： {total_all2all_single_time / total_time}')
print(f'allreduce_time占比： {total_allreduce_time / total_time}')
print(f'average_all2all_msg_size: {total_all2all_msg_size/299}')
print(f'average_allreduce_msg_size: {total_allreduce_msg_size/299}')
print(f'all2all_ratio: {total_all2all_msg_size/total_all2all_single_time}GB')
print(f'allreduce_ratio: {total_allreduce_msg_size/total_allreduce_time}')
print(f'average_all2all_time: {total_all2all_single_time/299}')
print(f'average_allreduce_time: {total_allreduce_time/299}')
print(f'average_iter_time: {total_time/299}')
print(f'sample/per sec: {960*299/total_time*1000}')
print(f'300Btoken cost time: {3e11/1024/(960*299/total_time*1000)/3600/24}')

