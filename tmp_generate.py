# -*- coding: utf-8 -*-
# @Time : 2022/8/17 09:57
# @Author : qingquan
# @File : tmp_generate.py
import yaml
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
load_path = '/raid/nlp/ckpts/ckpts_1.7b/transformers_200000'
special_token_file = '/raid/nlp/projects/nlp-megatron-deepspeed/ch_tokenizer_data/special_tokens.yaml'
special_tokens = open(special_token_file)
special_tokens = yaml.load(special_tokens, Loader=yaml.FullLoader)['special_tokens']
tokenizer = BertTokenizer.from_pretrained(load_path, additional_special_tokens=special_tokens)
model = GPT2LMHeadModel.from_pretrained(load_path)
print(model)
print(model.num_parameters())
exit()
# print(model.config)
text_generator = TextGenerationPipeline(model, tokenizer, device=7)
sen = '（玄幻体）一剑挥出，空中顿时出现了一柄巨大的剑光 。 接 着 一 股 气 流 冲 向 地 面 ， 对 准 了 正 要 落 地 休 息 的 一 位 英 雄 。 这 是 我 的 剑 ！ 英 雄 还 未 反 应 过 来 ， 已 然 被 落 地 的 一 剑 击 中 。'
sen = sen.replace(' ', '')

for i in range(5):
    res = text_generator(sen, max_length=200, do_sample=True)
    output = res[0]['generated_text'].replace(' ', '')
    print(output)
