#!/bin/bash
python deepspeed_to_transformers.py --input_folder /raid/nlp/ckpts/ckpts_1.7b/global_step20000 \
                                    --output_folder /raid/nlp/ckpts/ckpts_1.7b/transformers_20000 \
                                    --target_tp 1 \
                                    --target_pp 1 \
                                    --for_release