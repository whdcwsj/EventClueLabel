# encoding: utf-8
# @author: 王宇静轩
# @file: new_predicter_class.py
# @time: 2023/3/6 下午6:05


import warnings

warnings.filterwarnings("ignore")

import os
import sys

# cur_dir = os.getcwd()
sys.path.append("..")

import time
import numpy as np
import torch
from tqdm import tqdm
import argparse
from transformers import BertTokenizer

from model.model import ClassifierWithBert4Layer
from config.config import Config

flags = ['score', 'scale', 'influence', 'strength', 'degree']

label_score_map = {
    0: -5,
    1: -3,
    2: -1,
    3: 1,
    4: 3,
    5: 5,
}

weight_name = "../data/saved_dict/10_22_train/"
weight_dict = {
    0: "10_21_data_event_score10-22_18.00.ckpt",
    1: "10_21_data_event_scale10-22_18.00.ckpt",
    2: "10_21_data_event_influence10-22_18.00.ckpt",
    3: "10_21_data_event_strength10-23_10.01.ckpt",
    4: "10_21_data_event_degree10-23_10.01.ckpt"
}

def device_distribute(input_model_count):
    model_gpu_correspond = {}
    model_count = input_model_count
    gpu_count = torch.cuda.device_count()
    gpu_ids = [int(_) for _ in range(gpu_count)]
    gpu_devices = ["cuda:{}".format(_) for _ in gpu_ids if torch.cuda.is_available()]

    interval = model_count / gpu_count
    if gpu_count > 0:
        if interval <= 1:
            for i in range(model_count):
                model_gpu_correspond[i] = gpu_devices[i]
        else:
            cur_gpu = 0
            cur_interval = interval
            for i in range(model_count):
                if cur_interval >= (i + 1):
                    model_gpu_correspond[i] = gpu_devices[cur_gpu]
                else:
                    cur_interval += interval
                    cur_gpu += 1
                    model_gpu_correspond[i] = gpu_devices[cur_gpu]

    else:
        for i in range(model_count):
            model_gpu_correspond[i] = "cpu"

    return model_gpu_correspond


class NewEventCluePredictor:
    def __init__(self):
        self.config = Config(dataset='../data', name='my_predict')
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.model_num = 5
        self.correspond_list = device_distribute(input_model_count=self.model_num)

        self.gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_list = []
        self.advance_load_model()

    def advance_load_model(self):
        for i in range(self.model_num):
            cur_model = ClassifierWithBert4Layer(config=self.config, flag=flags[i]).to(
                torch.device(self.correspond_list[i]))
            cur_model.load_state_dict(torch.load(weight_name+weight_dict[i]))
            self.model_list.append(cur_model)

    def predictor(self, text, primaryClassification):
        if primaryClassification not in ["0", "1", "2", "3", "4"]:
            print("时间线索数据ID选择错误，请重新选择")
            return -100
        else:
            # 文本特征初始化
            tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
            tokenized = tokenizer(text=text, max_length=self.config.pad_size, truncation=True)
            token_ids = tokenized['input_ids']
            masks = tokenized['attention_mask']
            token_ids = torch.tensor(token_ids).to(self.gpu_device)
            masks = torch.tensor(masks).to(self.gpu_device)
            if primaryClassification == "2":
                scale = self.inference(model=self.model_list[1], token=token_ids, mask=masks, flag="scale")
                strength = self.inference(model=self.model_list[3], token=token_ids, mask=masks, flag="strength")
            else:
                scale = self.inference(model=self.model_list[2], token=token_ids, mask=masks, flag="influence")
                strength = self.inference(model=self.model_list[4], token=token_ids, mask=masks, flag="degree")
            score = self.inference(model=self.model_list[0], token=token_ids, mask=masks, flag="score")

            return scale, strength, score

    def inference(self, model, token, mask, flag):
        model.eval()
        cur_token = token.unsqueeze(0)
        cur_mask = mask.unsqueeze(0)
        with torch.no_grad():
            pred = model(cur_token, cur_mask)
            _, predict = torch.max(input=pred, dim=1)
            if torch.cuda.is_available():
                predict = predict.cpu()

        if flag == 'score':
            cur_id = predict.item()
            predict_id = label_score_map[cur_id]
        else:
            predict_id = predict.item() + 1

        return predict_id


if __name__ == '__main__':
    # primaryClassification=2
    # 对应的五个答案为: score 5, scale 3, influence 不存在, strength 3, degree不存在
    input_text1 = '俄罗斯总统新闻秘书佩斯科夫表示，北约已经事实上卷入俄乌冲突，但俄罗斯将把对乌克兰的“特别军事行动”进行到底。'

    # primaryClassification=4
    # 对应的五个答案为: score -3, scale 不存在, influence 1, strength 不存在, degree 1
    input_text2 = '德国最大租车公司西克斯特（Sixt）计划在今后6年内采购10万辆中国厂商比亚迪生产的电动汽车。'

    # primaryClassification=2
    # 对应的五个答案为: score -5, scale 2, influence 不存在, strength 2, degree 不存在
    input_text3 = '我空军轰-6轰炸机、空警-2000预警机、运-8电子干扰机、图-154电子侦察机以及苏-35、歼-11战机护航编队等组成两个打击集群，分别从台湾的南方和北方两个不同的方向，同时并进完成绕岛巡航'
    data_id = 0
    # flags = ['score', 'scale', 'influence', 'strength', 'degree']

    # 推理时加载模型
    # myclue = EventCluePredictor()
    # res = myclue.predictor(text=input_text1, flag_id=data_id)
    # print(f"返回的结果为：{res}")

    # 预先加载模型
    myclue = NewEventCluePredictor()
    res = myclue.predictor(text=input_text1, primaryClassification="2")
    print(res)
