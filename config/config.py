# encoding: utf-8
# @author: 王宇静轩
# @file: config.py
# @time: 2022/10/20 下午4:26

import time
import torch
from tensorboardX import SummaryWriter


class Config(object):
    """
    参数配置
    """

    def __init__(self, dataset='../data', name='wsj'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = name
        # 10-21数据
        # 1、事件打分(6分类）
        self.train_score_path = dataset + '/processed_data' + '/event_score_1_3_5.csv'
        self.num_score_classes = 6
        self.score_every_class = [23, 71, 140, 380, 513, 335]
        # 2、事件规模(3分类)
        self.train_scale_path = dataset + '/processed_data' + '/event_scale.csv'
        self.num_scale_classes = 3
        self.scale_every_class = [564, 169, 88]
        # 3、事件影响(2分类)
        self.train_influence_path = dataset + '/processed_data' + '/event_influence.csv'
        self.num_influence_classes = 2
        self.influence_every_class = [652, 943]
        # 4、事件强度(3分类)
        self.train_strength_path = dataset + '/processed_data' + '/event_strength.csv'
        self.num_strength_classes = 3
        self.strength_every_class = [518, 184, 116]
        # 5、事件程度(4分类)
        self.train_degree_path = dataset + '/processed_data' + '/event_degree.csv'
        self.num_degree_classes = 4
        self.degree_every_class = [1313, 181, 99, 17]

        # 训练存储相关参数
        self.bert_path = dataset + '/bert_chinese'
        self.pad_size = 100
        self.log_path = dataset + '/log/' + self.model_name
        self.save_model_path = dataset + '/saved_dict/' + self.model_name + time.strftime('%m-%d_%H.%M',
                                                                                          time.localtime()) + '.ckpt'

        # 模型的相关参数
        self.embedding_dim = 768
        self.hidden_size = 512
        self.lstm_layers = 2
        self.dropout = 0.1

        # 模型训练相关参数
        self.batch_size = 32
        self.learning_rate = 1e-5
        self.num_epochs = 150
        self.require_improvement = 50
