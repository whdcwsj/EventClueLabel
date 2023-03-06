# encoding: utf-8
# @author: 王宇静轩
# @file: dataloader.py
# @time: 2022/10/20 下午7:31

import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from config.config import Config


def load_data(dir_path):
    df = pd.read_csv(dir_path, sep=',')
    return df


def split_dataset_pd(dataset, seed, threshold=0.2):
    train_data = dataset.sample(frac=1 - threshold, random_state=seed, axis=0)
    dev_data = dataset[~dataset.index.isin(train_data.index)]
    return train_data, dev_data


class MyDataset(Dataset):
    def __init__(self, config, dataset, test=False):
        super(MyDataset, self).__init__()
        self.config = config
        self.device = self.config.device
        self.test = test
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.dataset = dataset
        # iloc函数通过行号取数据
        # 第一列文本，第二列标签
        # np.asarray除非必要，否则不会copy该对象;np.array(默认)将会copy该对象
        self.text_arr = np.asarray(self.dataset.iloc[:, 0])
        if self.test is False:
            self.label = np.asarray(self.dataset.iloc[:, 1])

    def __getitem__(self, item):
        text_ids = self.text_arr[item]
        tokenized = self.tokenizer(text=text_ids, max_length=self.config.pad_size, truncation=True)
        token_ids = tokenized['input_ids']
        masks = tokenized['attention_mask']
        # 补[0]
        padding_size = self.config.pad_size - len(token_ids)
        if padding_size >= 0:
            token_ids = token_ids + [0] * padding_size
            masks = masks + [0] * padding_size
        else:
            token_ids = token_ids[:self.config.pad_size]
            masks = masks[:self.config.pad_size]
        # 张量化
        token_ids = torch.tensor(token_ids).to(self.device)
        masks = torch.tensor(masks).to(self.device)
        if self.test is False:
            cur_label = self.label[item]
            labels = torch.tensor(cur_label-1).to(self.device)
            return token_ids, masks, labels
        else:
            return token_ids, masks

    def __len__(self):
        return len(self.dataset)

    # 重要性采样需要用到
    def get_all_classes(self):
        return self.label - 1


if __name__ == '__main__':
    config = Config(dataset='../data', name='wsj')
    # all_set = load_data(dir_path=config.train_score_path)
    # all_set = load_data(dir_path=config.train_scale_path)
    # all_set = load_data(dir_path=config.train_influence_path)
    # all_set = load_data(dir_path=config.train_strength_path)
    all_set = load_data(dir_path=config.train_degree_path)
    train_dataset, dev_dataset = split_dataset_pd(dataset=all_set, seed=2)
    # 统计训练集和验证集的数据分布
    train_count = train_dataset.groupby(['degree'], as_index=False)['degree'].agg({'cnt': 'count'})
    # dev_count = dev_dataset.groupby(['score'], as_index=False)['score'].agg({'cnt': 'count'})
    print(train_count)
    dataset = MyDataset(config=config, dataset=train_dataset)
    # print(dataset.get_all_classes())
    # print(type(dataset.get_all_classes()))
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    for iter, data in enumerate(dataloader):
        print(f"{iter}:{data}")
