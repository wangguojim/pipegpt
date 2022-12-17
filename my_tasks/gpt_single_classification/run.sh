#!/bin/bash
ckpt_path='../../ckpts'

# wiki
#nohup python trainer.py --data_path  "data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx" \
#                        --device  "cuda:0" \
#                        --base_params "$ckpt_path/transformers_25000" \
#                        --special_token_file "special_tokens.yaml" \
#                        --save_path "$ckpt_path/small_net_wiki" \
#                        --tensorboard_path "$ckpt_path/small_net_wiki/tensorboard_dir" \
#			                  --lr 1e-4 &


# random
#nohup python trainer.py --data_path  "data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx" \
#                        --device  "cuda:2" \
#                        --base_params "$ckpt_path/transformers_25000" \
#                        --special_token_file "special_tokens.yaml" \
#                        --save_path "$ckpt_path/small_net_random" \
#                        --tensorboard_path "$ckpt_path/small_net_random/tensorboard_dir" \
#                        --lr 1e-4 &

# cmcc 7000
#nohup python trainer.py --data_path  "data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx" \
#                        --device  "cuda:1" \
#                        --base_params "$ckpt_path/cmcc_transformers_7000" \
#                        --special_token_file "special_tokens.yaml" \
#                        --save_path "$ckpt_path/small_net_cmcc" \
#                        --tensorboard_path "$ckpt_path/small_net_cmcc/tensorboard_dir" \
#                        --lr 1e-4 &

# cmcc 18000
nohup python trainer.py --data_path  "data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx" \
                        --device  "cuda:3" \
                        --base_params "$ckpt_path/cmcc_transformers_18000" \
                        --special_token_file "special_tokens.yaml" \
                        --save_path "$ckpt_path/small_net_cmcc_18000" \
                        --tensorboard_path "$ckpt_path/small_net_cmcc_18000/tensorboard_dir" \
                        --lr 1e-4 &