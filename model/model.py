# encoding: utf-8
# @author: 王宇静轩
# @file: model.py
# @time: 2022/10/20 下午9:06


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertModel, BertConfig, AutoConfig
from config.config import Config
from train.dataloader import load_data, split_dataset_pd, MyDataset
import logging


class ClassifierWithBert4Layer(nn.Module):
    def __init__(self, config, flag):
        super(ClassifierWithBert4Layer, self).__init__()
        self.config = config
        bert_config = BertConfig.from_pretrained(self.config.bert_path)
        # 获取每层的输出
        bert_config.update({'output_hidden_states': True})
        # self.bert = BertModel.from_pretrained(self.config.bert_path, config=bert_config)

        try:
            self.bert = BertModel.from_pretrained(self.config.bert_path, config=bert_config)
        except Exception as e:
            logging.warning(e)
            # bert_config_temp = AutoConfig.from_pretrained(self.config.bert_path)
            self.bert = BertModel(config=bert_config)

        self.lstm = nn.LSTM(self.config.embedding_dim * 4, self.config.hidden_size, self.config.lstm_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=self.config.dropout)
        self.linear1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)
        self.linear2 = nn.Linear(self.config.hidden_size * 2, 1)
        self.dropout = nn.Dropout(self.config.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.linear3 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)
        if flag == 'score':
            self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_score_classes)
        elif flag == 'scale':
            self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_scale_classes)
        elif flag == 'influence':
            self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_influence_classes)
        elif flag == 'strength':
            self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_strength_classes)
        elif flag == 'degree':
            self.classifier = nn.Linear(self.config.hidden_size * 2, self.config.num_degree_classes)

    def forward(self, token_id, mask):
        outputs = self.bert(token_id, attention_mask=mask)
        # last_hidden_state, pooler_output, hidden_states
        all_hidden_states = torch.stack(outputs[2])  # (13,batch,seq,768)
        concat_last_4layers = torch.cat((all_hidden_states[-1],
                                         all_hidden_states[-2],
                                         all_hidden_states[-3],
                                         all_hidden_states[-4]), dim=-1)  # (batch,seq,768*4)
        H, _ = self.lstm(concat_last_4layers)  # (batch,seq,hidden*2)
        # 自注意力
        self_attention = torch.tanh(self.linear1(self.dropout(H)))  # (batch,seq,hidden*2)
        # squeeze()删除维度为1
        self_attention = self.linear2(self.dropout(self_attention)).squeeze()  # (batch,seq)
        self_attention = self_attention + -10000 * (mask == 0).float()  # (batch,seq)
        self_attention = self.softmax(self_attention)  # (batch,seq)
        # 加入自注意力，得到句子初始特征
        sent_encoding = torch.sum(H * self_attention.unsqueeze(-1), dim=1)  # (batch,hidden*2)
        pre_pred = torch.tanh(self.linear3(self.dropout(sent_encoding)))  # (batch,hidden*2)
        pred = self.classifier(self.dropout(pre_pred))  # (batch,num_classes)

        return pred


if __name__ == '__main__':
    config = Config(dataset='../data', name='wsj')
    all_set = load_data(dir_path=config.train_scale_path)
    train_dataset = MyDataset(config=config, dataset=all_set)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    model = ClassifierWithBert4Layer(config=config, flag='scale').to(config.device)
    for inputs in train_dataloader:
        token_ids, masks, labels = inputs
        model.zero_grad()
        pred = model(token_ids, masks)
