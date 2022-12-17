# -*- coding: utf-8 -*-
# @Time : 2022/10/20 18:46
# @Author : qingquan
# @File : cal_params.py
import torch


def cal_dense_params(n_layers, hidden, max_len=1024, vocab_size=21248):
    # return vocab_size*hidden*2+max_len*hidden+(hidden*13+max_len*max_len+hidden*hidden*12)*n_layers+2*hidden
    return vocab_size * hidden + max_len * hidden + (hidden * 13 + hidden * hidden * 12) * n_layers + 2 * hidden


def cal_moe_params(n_layers, n_experts, hidden, max_len=1024, vocab_size=21248):
    return int(vocab_size * hidden + max_len * hidden + \
           (hidden * 13 + hidden ** 2 * 12) * n_layers / 2 + \
           ((hidden * 8 + hidden ** 2 * 4 + n_experts * hidden) + (hidden * 5 + hidden ** 2 * 8) * n_experts) * n_layers / 2 + \
           2 * hidden)


def cal_one_stage_params(hidden, max_len):
    return hidden * 13 + max_len * max_len + hidden * hidden * 12


def sum_params(params):
    total = 0
    one_layer_total = 0
    for k, v in params.items():
        print(k, v.shape)
        if len(v.shape) == 1:
            total += v.shape[0]
        elif len(v.shape) >= 2:
            total += v.shape[-1] * v.shape[-2]
    #     if '22' in k:
    #         print(k, v.shape)
    #         if len(v.shape) == 1:
    #             one_layer_total += v.shape[0]
    #         elif len(v.shape) >= 2:
    #             one_layer_total += v.shape[-1] * v.shape[-2]
    # return total, one_layer_total
    return total


if __name__ == '__main__':
    # print(cal_one_stage_params(hidden=2304, max_len=1024))
    # exit()
    hidden=2048
    layers=22
    n_experts = 24

    print(cal_dense_params(layers, hidden, max_len=1024))
    print(cal_moe_params(layers, n_experts, hidden, max_len=1024))
    exit()
    ### 计算transformer gpt model的参数量
    # 64779520
    print(sum_params('/raid/nlp/ckpts/ckpts_1.7b/transformers_200000/pytorch_model.bin'))
    # exit()
    ###计算deepspeed gpt model参数量
    total = 0
    for i in range(1, 29):
        try:
            stage_total = sum_params(f'/raid/nlp/ckpts/ckpts_1.7b/global_step200000/layer_{str(i).zfill(2)}-model_00-model_states.pt')
            # if i == 3:
                # print(stage_total)
                # print(stage_total+1024**2)
            total += stage_total
        except:
            pass
            # print(i)
    print(total)
    # print(total+1024*1024*24)
    exit()

    ###观察deepspeed gpt model参数量
    for i in range(1, 29):
        try:
            params = torch.load(
                f'/raid/nlp/ckpts/ckpts_1.7b/global_step200000/layer_{str(i).zfill(2)}-model_00-model_states.pt')
            for k, v in params.items():
                print(k, v.shape)
            print('-' * 100)
        except:
            pass

        # input_layernorm.weight torch.Size([2304]) 1
        # input_layernorm.bias torch.Size([2304]) 2
        # attention.query_key_value.weight torch.Size([6912, 2304]) 3
        # attention.query_key_value.bias torch.Size([6912]) 4
        # attention.dense.weight torch.Size([2304, 2304]) 5
        # attention.dense.bias torch.Size([2304]) 6
        # post_attention_layernorm.weight torch.Size([2304]) 7
        # post_attention_layernorm.bias torch.Size([2304]) 8
        # mlp.dense_h_to_4h.weight torch.Size([9216, 2304]) 9
        # mlp.dense_h_to_4h.bias torch.Size([9216]) 10
        # mlp.dense_4h_to_h.weight torch.Size([2304, 9216])  11
        # mlp.dense_4h_to_h.bias torch.Size([2304]) 12

        # transformer.h.23.ln_1.weight torch.Size([2304]) 1
        # transformer.h.23.ln_1.bias torch.Size([2304]) 2
        # transformer.h.23.attn.bias torch.Size([1, 1, 1024, 1024])
        # transformer.h.23.attn.masked_bias torch.Size([])
        # transformer.h.23.attn.c_attn.weight torch.Size([2304, 6912]) 3
        # transformer.h.23.attn.c_attn.bias torch.Size([6912]) 4
        # transformer.h.23.attn.c_proj.weight torch.Size([2304, 2304]) 5
        # transformer.h.23.attn.c_proj.bias torch.Size([2304]) 6
        # transformer.h.23.ln_2.weight torch.Size([2304]) 7
        # transformer.h.23.ln_2.bias torch.Size([2304]) 8
        # transformer.h.23.mlp.c_fc.weight torch.Size([2304, 9216]) 9
        # transformer.h.23.mlp.c_fc.bias torch.Size([9216])10
        # transformer.h.23.mlp.c_proj.weight torch.Size([9216, 2304]) 11
        # transformer.h.23.mlp.c_proj.bias torch.Size([2304]) 12