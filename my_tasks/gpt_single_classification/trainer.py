# -*- coding: utf-8 -*-
# @Time : 2022/5/9 20:53
# @Author : qingquan
# @File : trainer.py
import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from config import get_config
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from process import MyDataset, DataLoader, paired_collate_fn
from log import Logger
import warnings
warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.pad_token_id = 0
        # gpt_config = GPT2Config(vocab_size=21248,
        #                         n_positions=2048,
        #                         n_embd=2304,
        #                         n_layer=24,
        #                         n_head=32,
        #                         n_inner=4,
        #                         )
        # self.gpt = GPT2Model(gpt_config)
        self.gpt = GPT2Model.from_pretrained(args.base_params)
        self.small_net = nn.Sequential(
            nn.Linear(2304, 2304),
            nn.ReLU(),
            nn.Linear(2304, 9)
        )
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.loss_fun = CrossEntropyLoss().to(config.device)

    def forward(self, enc_input_padded, enc_mask, labels=None):
        # with torch.no_grad():
        hidden_states = self.gpt(input_ids=enc_input_padded, attention_mask=enc_mask).last_hidden_state
        batch_size = enc_input_padded.shape[0]
        sequence_lengths = torch.ne(enc_input_padded, self.pad_token_id).sum(-1) - 1
        pooled_logits = hidden_states[range(batch_size), sequence_lengths]
        logits = self.small_net(pooled_logits)
        if labels is not None:
            loss = self.loss_fun(logits, labels)
            return logits, loss
        else:
            return logits, None


class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(console=True, log_path=config.save_path).logger
        self.model = Model(config).to(config.device)
        # self.model = GPT2ForSequenceClassification.from_pretrained('../../ckpts/transformers_25000/', num_labels=9,
        #                                                            problem_type='single_label_classification').to(config.device)
        # self.model.config.pad_token_id = 0
        self.scheduler = ''
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss().to(config.device)
        self.writer = SummaryWriter(config.tensorboard_path)

    def _update(self, inputs, mode):
        inputs = [i.to(self.config.device)if not isinstance(i, list) else i for i in inputs ]
        enc_input_padded, enc_mask, labels = inputs
        labels = torch.Tensor(labels).long().to(self.config.device)
        outputs, loss = self.model(enc_input_padded, enc_mask, labels=labels)
        # out = self.model(input_ids=enc_input_padded, attention_mask=enc_mask, labels=labels)
        # outputs, loss = out.logits, out.loss
        num_corr, num = self._cal_preformance(outputs, labels)
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # for p in self.model.gpt.parameters():
        #     print(p)
        #     break
        # for p in self.model.small_net.parameters():
        #     print(p)
        #     break
        return loss.item(), num_corr, num

    def _cal_preformance(self, logits, labels):
        # mask = mask.unsqueeze(dim=-1).repeat(1, 1, logits.shape[-1])
        # logits = logits.masked_select(mask).view(-1, logits.shape[-1])
        # logits = logits.reshape(-1)
        # labels = torch.Tensor(labels).to(self.config.device)
        # loss = self.model(logits, labels=labels).loss
        # loss = self.criterion(logits, labels)
        num_corr = 0
#         for i in range(logits.shape[0]):
#             if ((logits[0] > 0.5) == labels[i]).all():
#                 num_corr += 1
        _, indices = logits.max(dim=1)
        num_corr = indices.eq(labels).sum().item()
        return num_corr, logits.shape[0]

    def _run_epoch(self, dataset, mode):
        if mode not in ['train', 'eval']:
            raise Exception("you must select 'train' or 'eval' as a value of mode!")
        total_corr = 0  # 预测正确总个数
        total_labs = 0  # 总标签数
        total_loss = 0  # 总损失
        total_num = 0  # 样本总个数
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        for idx, inputs in tqdm(enumerate(dataset), total=len(dataset), desc=mode):
            # if idx == 29: continue
            # print(idx)

            loss, num_corr, num = self._update(inputs, mode)

            total_corr += num_corr
            total_labs += num
            total_loss += loss
            total_num += 1

            # if idx == 33: exit()
        avg_loss = round(total_loss/(total_num), 5)  # 平均损失
        accura = round(total_corr/(total_labs), 4)  # 准确率
        # f1 = round(2*recall*accura/(recall+accura+1e-5), 2)  # f1
        return avg_loss, accura

    def train(self, num_eopch, train_dataset, val_dataset, save_path, checkpoint_path=False):
        start_epoch = 0
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
            start_epoch = checkpoint['epoch']
            assert start_epoch < num_eopch
            self.model.load_state_dict(checkpoint['params'])

        t_avg_loss, t_accura = self._run_epoch(train_dataset, 'eval')
        e_avg_loss, e_accura = self._run_epoch(val_dataset, 'eval')
        self.writer.add_scalars('show', {'train-loss': t_avg_loss,
                                               'eval-loss': e_avg_loss,
                                               'train-acc': t_accura,
                                               'eval-acc': e_accura}
                                       , 1)

        self.logger.info(f'-Train  loss:{t_avg_loss}   accuracy:{t_accura * 100}%')
        self.logger.info(f'-Eval   loss:{e_avg_loss}   accuracy:{e_accura * 100}%')

        best_params, save_epoch = 0, 0
        for epoch in range(start_epoch, num_eopch):
            self.logger.info(f'Epoch {epoch+1}/{num_eopch}:')
            t_avg_loss, t_accura = self._run_epoch(train_dataset, 'train')
            e_avg_loss, e_accura = self._run_epoch(val_dataset, 'eval')
            self.writer.add_scalars('show', {'train-loss': t_avg_loss,
                                               'eval-loss': e_avg_loss,
                                               'train-acc': t_accura,
                                               'eval-acc': e_accura}
                                       , epoch+2)
            self.logger.info(f'-Train  loss:{t_avg_loss}   accuracy:{t_accura*100}%')
            self.logger.info(f'-Eval   loss:{e_avg_loss}   accuracy:{e_accura*100}%')
            if abs(t_accura-e_accura) <= 0.1 and t_accura >= 0 and e_accura >= 0:
                if e_accura>best_params:
                    best_params = e_accura
                    model_state_dict = self.model.state_dict()
                    checkpoint = {
                        'params': model_state_dict,
                        'configs': self.config,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch}
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    if os.path.exists(save_path + '/' + str(save_epoch) + '.pth'):
                        os.remove(save_path + '/' + str(save_epoch) + '.pth')
                        self.logger.info(f'the model saved in: {save_path}/{save_epoch}.pth  will be revomed')
                    file_name = f'{save_path}/{epoch}.pth'
                    self.logger.info(f'the model will be saved in: {file_name}')
                    torch.save(checkpoint, file_name)
                    save_epoch = epoch

    def test(self, test_dataset, checkpoint_path=False):
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['params'])
        avg_loss, accura = self._run_epoch(test_dataset, 'eval')
        self.logger.info(f'-test   loss:{avg_loss}   accuracy:{accura * 100}%')


if __name__ == '__main__':
    config = get_config()
    # 定义Summary_Writer
    print(config)
    train_dataset = MyDataset(config.data_path, mode='train')
    eval_dataset = MyDataset(config.data_path, mode='test')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=paired_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=paired_collate_fn)
    trainer = Trainer(config)
    trainer.train(config.num_epoch, train_loader, eval_loader, config.save_path,)
                      # checkpoint_path='./data/ckpts/fine_tuned/30.pth')
    # trainer.test(eval_loader, checkpoint_path='./ckpts/finetune2/7.pth')

    # --data_path  data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx
    # --device  cuda:0
    # --base_params  ../../ckpts/transformers_25000
    # --special_token_file special_tokens.yaml
    # --save_path ./ckpts/
    # --tensorboard_path ./ckpts/tensorboard_dir

    # nohup python trainer.py --data_path  data/营业厅数字人项目事项v1.0-人工匹配规则枚举.xlsx --device  cuda:0 \
    #                         --base_params.. /../ ckpts / transformers_25000 \
    #                         --special_token_file special_tokens.yaml \
    #                         --save_path. / ckpts/ \
    #                         --tensorboard_path. /ckpts/tensorboard_dir
