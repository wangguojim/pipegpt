#!/bin/bash

python /raid/nlp/projects/nlp-megatron-deepspeed/tools/preprocess_chinese_data.py \
       --input /raid/nlp/data/aggregated_meta_data/dataset_10b_12.txt \
       --output-prefix /raid/nlp/data/aggregated_meta_data/dataset_10b_12 \
       --json-keys 'text' \
       --vocab /raid/nlp/projects/nlp-megatron-deepspeed/ch_tokenizer_data/vocab.txt \
       --dataset-impl mmap \
       --workers 4 \
       --special-token-file /raid/nlp/projects/nlp-megatron-deepspeed/ch_tokenizer_data/special_tokens.yaml
