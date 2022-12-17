# -*- coding: utf-8 -*-
# @Time : 2022/7/22 00:14
# @Author : qingquan
# @File : aggregate_data.py
import os
import json
from tqdm import tqdm
import shutil

prefix = 'dataset_10b_'
data_path = f'/raid/nlp/data/meta_data/'  # 原始数据集位置
aggregated_path = f'/raid/nlp/data/aggregated_meta_data'  # 聚合后数据集位置
all_file_names = [os.path.join(root, f) for root, dirs, files in os.walk(data_path) for f in files if f.endswith('.txt')
                  and 'txt_files' in os.path.join(root, f)]  # 递归遍历目录下文件
# print(len(all_file_names))
# exit()
# 运行脚本前清空文件夹
try:
    shutil.rmtree(aggregated_path)
except:
    pass
# exit()
os.mkdir(aggregated_path)   # 创建文件夹


def byte_size(s):
    """统计字符串大小"""
    return len(s.encode('utf-8'))

real_total_size = 0  # 统计所有文件的大小
total_size = 0  # 统计所有文件有用内容的大小
threshold = 20*1024*1024*1024  # 预设定文件大小20G
on_time_size = 0  # 实时文件大小
# finished_read_file_flag = True   # 默认为有文件读取错误，以便可读取第一个文件
write_file_count = 0
wf = open(f'{aggregated_path}/{prefix}{write_file_count}.txt', 'w')  # 打开第一个要写入的文件

for processing_read_file_count, file_name in tqdm(enumerate(all_file_names),
                                                  desc=f'正在处理个文件',
                                                  total=len(all_file_names)):
    rf = open(file_name, 'r')  # 读取的新文件文件
    for json_data in rf:
        json_data = json.loads(json_data)
        size = byte_size(json_data['text'])  # 字节数
        on_time_size += size
        total_size += size
        if on_time_size < threshold:
            wf.write(json.dumps(json_data, ensure_ascii=False) + '\n')
        else:
            wf.close()  # 关闭当前要写入的文件
            real_total_size += os.path.getsize(f'{aggregated_path}/{prefix}{write_file_count}.txt')
            write_file_count += 1  # 下一个文件序号
            print(f', 已完成第{write_file_count}个文件的写入,大小为{on_time_size / 1024 / 1024}MB')
            wf = open(f'{aggregated_path}/{prefix}{write_file_count}.txt', 'w')  # 打开下一个要写入的文件
            on_time_size = 0  # 重置实时文件大小
            wf.write(json.dumps(json_data, ensure_ascii=False) + '\n')
    rf.close()
wf.close()  # 关闭最后一个要写入的文件
real_total_size += os.path.getsize(f'{aggregated_path}/{prefix}{write_file_count}.txt')
print(f'有用信息大小为:{total_size/1024/1024/1024}GB')
print(f'所有文件大小为:{real_total_size/1024/1024/1024}GB')
