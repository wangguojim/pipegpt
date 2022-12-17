# -*- coding: utf-8 -*-
# @Time : 2022/5/11 19:43
# @Author : qingquan
# @File : run.py
from deepspeed.launcher.runner import main

if __name__ == '__main__':
    main()


# --hostfile
# hostfile
# /raid/nlp/nlp-megatron-deepspeed/pretrain_gpt.py
# --tensor-model-parallel-size
# 1
# --pipeline-model-parallel-size
# 4
# --num-layers
# 24
# --hidden-size
# 2304
# --num-attention-heads
# 32
# --seq-length
# 2048
# --loss-scale
# 12
# --max-position-embeddings
# 2048
# --micro-batch-size
# 1
# --global-batch-size
# 20
# --train-iters
# 50000
# --lr
# 1.0e-5
# --min-lr
# 6.0e-6
# --lr-decay-style
# cosine
# --log-interval
# 1
# --eval-iters
# 40
# --eval-interval
# 1000
# --data-path
# 1
# /raid/nlp/processed_data/zhiyuan/part-2021278643_content_document
# 1
# /raid/nlp/processed_data/THUCNews/股票/647234_text_document
# --vocab-file
# /raid/nlp/nlp-megatron-deepspeed/ch_tokenizer_data/vocab.txt
# --special-token-file
# /raid/nlp/nlp-megatron-deepspeed/ch_tokenizer_data/special_tokens.yaml
# --save
# ckpts
# --load
# ckpts
# --save-interval
# 1000
# --split
# 950,49,1
# --clip-grad
# 1.0
# --weight-decay
# 0.1
# --adam-beta1
# 0.9
# --adam-beta2
# 0.95
# --init-method-std
# 0.006
# --tensorboard-dir
# tensorboard-dir
# --fp16
# --checkpoint-activations
# --deepspeed
# --deepspeed_config=/raid/nlp/nlp-megatron-deepspeed/ds_config.json
# --zero-stage=0
# --deepspeed-activation-checkpointing