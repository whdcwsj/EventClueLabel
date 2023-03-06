# encoding: utf-8
# @author: 王宇静轩
# @file: predicter.py
# @time: 2022/10/23 下午8:37

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

from event_clue_label.model.model import ClassifierWithBert4Layer
from event_clue_label.config.config import Config

flags = ['score', 'scale', 'influence', 'strength', 'degree']

label_score_map = {
    0: -5,
    1: -3,
    2: -1,
    3: 1,
    4: 3,
    5: 5,
}


def inference(model, token, mask, flag):
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


def event_clue_predict(text, flag_id):
    config = Config(dataset='/app/biaozhu/lrk/event_classification/refined/data', name='my_predict')
    # 输入文本处理
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    tokenized = tokenizer(text=text, max_length=config.pad_size, truncation=True)
    token_ids = tokenized['input_ids']
    masks = tokenized['attention_mask']
    token_ids = torch.tensor(token_ids).to('cpu')
    masks = torch.tensor(masks).to('cpu')

    data_flag = flags[flag_id]
    model = ClassifierWithBert4Layer(config=config, flag=data_flag).to('cpu')

    if data_flag == 'score':
        model.load_state_dict(torch.load(r"../data/saved_dict/10_22_train/10_21_data_event_score10-22_18.00.ckpt", map_location=torch.device('cpu')))
    elif data_flag == 'scale':
        model.load_state_dict(torch.load(r"../data/saved_dict/10_22_train/10_21_data_event_scale10-22_18.00.ckpt", map_location=torch.device('cpu')))
    elif data_flag == 'influence':
        model.load_state_dict(torch.load(r"../data/saved_dict/10_22_train/10_21_data_event_influence10-22_18.00.ckpt", map_location=torch.device('cpu')))
    elif data_flag == 'strength':
        model.load_state_dict(torch.load(r"../data/saved_dict/10_22_train/10_21_data_event_strength10-23_10.01.ckpt", map_location=torch.device('cpu')))
    elif data_flag == 'degree':
        model.load_state_dict(torch.load(r"../data/saved_dict/10_22_train/10_21_data_event_degree10-23_10.01.ckpt", map_location=torch.device('cpu')))
    else:
        print("输入的data_flag错误！")
        return

    result = inference(model=model, token=token_ids, mask=masks, flag=data_flag)

    return result


def interface_event_clue_predict(text, primaryClassification):
    config = Config(dataset='/app/biaozhu/lrk/event_classification/refined/data', name='my_predict')
    # 输入文本处理
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    tokenized = tokenizer(text=text, max_length=config.pad_size, truncation=True)
    token_ids = tokenized['input_ids']
    masks = tokenized['attention_mask']
    token_ids = torch.tensor(token_ids).to('cpu')
    masks = torch.tensor(masks).to('cpu')
    if primaryClassification =="2":
        model = ClassifierWithBert4Layer(config=config, flag="scale").to('cpu')
        model.load_state_dict(torch.load(r"/app/biaozhu/lrk/event_classification/refined/data/saved_dict/10_22_train/10_21_data_event_scale10-22_18.00.ckpt", map_location=torch.device('cpu')))
        scale = inference(model=model, token=token_ids, mask=masks, flag="scale")

        model1 = ClassifierWithBert4Layer(config=config, flag="strength").to('cpu')
        model1.load_state_dict(torch.load(r"/app/biaozhu/lrk/event_classification/refined/data/saved_dict/10_22_train/10_21_data_event_strength10-23_10.01.ckpt", map_location=torch.device('cpu')))
        strength = inference(model=model1, token=token_ids, mask=masks, flag="strength")
    else:
        model = ClassifierWithBert4Layer(config=config, flag="influence").to('cpu')
        model.load_state_dict(torch.load(r"/app/biaozhu/lrk/event_classification/refined/data/saved_dict/10_22_train/10_21_data_event_influence10-22_18.00.ckpt", map_location=torch.device('cpu')))
        scale = inference(model=model, token=token_ids,
                           mask=masks, flag="influence")

        model1 = ClassifierWithBert4Layer(config=config, flag="degree").to('cpu')       
        model1.load_state_dict(torch.load(r"/app/biaozhu/lrk/event_classification/refined/data/saved_dict/10_22_train/10_21_data_event_degree10-23_10.01.ckpt", map_location=torch.device('cpu')))          
        strength = inference(model=model1, token=token_ids,
                           mask=masks, flag="degree")
    model2 = ClassifierWithBert4Layer(config=config, flag="score").to('cpu')
    model2.load_state_dict(torch.load(r"/app/biaozhu/lrk/event_classification/refined/data/saved_dict/10_22_train/10_21_data_event_score10-22_18.00.ckpt", map_location=torch.device('cpu')))
    score = inference(model=model2, token=token_ids, mask=masks, flag="score")

    return scale, strength, score


if __name__ == '__main__':
    # 实际label为6
    input_text1 = '俄罗斯总统新闻秘书佩斯科夫表示，北约已经事实上卷入俄乌冲突，但俄罗斯将把对乌克兰的“特别军事行动”进行到底。'
    # 实际label为2
    input_text2 = '德国最大租车公司西克斯特（Sixt）计划在今后6年内采购10万辆中国厂商比亚迪生产的电动汽车。'
    # 实际label为1
    input_text3 = '我空军轰-6轰炸机、空警-2000预警机、运-8电子干扰机、图-154电子侦察机以及苏-35、歼-11战机护航编队等组成两个打击集群，分别从台湾的南方和北方两个不同的方向，同时并进完成绕岛巡航'
    data_id = 1
    # flags = ['score', 'scale', 'influence', 'strength', 'degree']
    if data_id not in [0, 1, 2, 3, 4]:
        print("时间线索数据ID选择错误，请重新选择")
    else:
        res_id = event_clue_predict(text=input_text3, flag_id=data_id)
        print(f"当前预测的event {flags[data_id]}，类别ID为：{res_id}")
