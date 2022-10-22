# encoding: utf-8
# @author: 王宇静轩
# @file: trainer.py
# @time: 2022/10/21 上午9:54

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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_recall_fscore_support
import argparse

from model.model import ClassifierWithBert4Layer
from config.config import Config
from dataloader import load_data, split_dataset_pd, MyDataset


# 数据不平衡，以Micro-F1为基准进行评判
def mytrain(config, model, train_dataset, dev_dataset, writer):
    # (precision, recall, micro_f1, _), macro_f1, dev_loss = myevaluate(config=config, model=model,
    #                                                                   dev_dataset=dev_dataset)
    # return
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    pre_best_performance = 0
    improved_epoch = 0
    for epoch in range(config.num_epochs):
        total_loss = 0
        model.train()
        data = tqdm(train_dataset, leave=True)
        for inputs in data:
            token_ids, masks, labels = inputs
            model.zero_grad()
            pred = model(token_ids, masks)
            # input-(N,C) C是类别的个数
            # target-(N)
            loss = F.cross_entropy(input=pred, target=labels)
            total_loss += loss.item()  # 返回原数值，跳出张量
            loss.backward()
            optimizer.step()
            # 设置进度条信息
            data.set_description(f"Epoch {epoch}")
            data.set_postfix(loss=loss.item())

        (precision, recall, micro_f1, _), macro_f1, dev_loss = myevaluate(config=config, model=model,
                                                                          dev_dataset=dev_dataset)
        writer.add_scalar(tag='loss/train', scalar_value=total_loss / len(train_dataset), global_step=epoch)
        writer.add_scalar(tag='loss/dev', scalar_value=dev_loss / len(dev_dataset), global_step=epoch)
        writer.add_scalars(main_tag='performance/f1', tag_scalar_dict={'micro_f1': micro_f1, 'macro_f1': macro_f1},
                           global_step=epoch)
        writer.add_scalar(tag='performance/precision', scalar_value=precision, global_step=epoch)
        writer.add_scalar(tag='performance/recall', scalar_value=recall, global_step=epoch)
        if pre_best_performance < micro_f1:
            pre_best_performance = micro_f1
            improved_epoch = epoch
            torch.save(model.state_dict(), config.save_model_path)
            print("model saved!!!")
        elif epoch - improved_epoch >= config.require_improvement:
            print("model didn't improve for a long time!So break!!!")
            break
    writer.close()


def myevaluate(config, model, dev_dataset):
    y_true, y_pred = [], []
    model.eval()
    evaluate_loss = 0
    with torch.no_grad():
        for data in tqdm(dev_dataset):
            token_ids, masks, labels = data
            model.zero_grad()
            pred = model(token_ids, masks)
            loss = F.cross_entropy(input=pred, target=labels)
            evaluate_loss += loss.item()
            # dim=1，其它与输入形状保持一致
            # 返回最大值，并且返回索引
            _, predict = torch.max(input=pred, dim=1)
            # GPU上的tensor不能直接转为numpy
            if torch.cuda.is_available():
                predict = predict.cpu()
                labels = labels.cpu()
            y_pred += predict.numpy().tolist()
            y_true += labels.numpy().tolist()

    macro_scores = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
    micro_scores = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro')
    print(f"Micro-F1: {micro_scores[2]}\n")
    print("Classification Report \n", classification_report(y_true=y_true, y_pred=y_pred, digits=4))

    return micro_scores, macro_scores[2], evaluate_loss


def main(data_flag):
    time_start = time.time()
    cur_name = '10_21_data_event_' + data_flag
    config = Config(dataset='../data', name=cur_name)
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    # 选择数据集
    if data_flag == 'score':
        cur_path = config.train_score_path
    elif data_flag == 'scale':
        cur_path = config.train_scale_path
    elif data_flag == 'influence':
        cur_path = config.train_influence_path
    elif data_flag == 'strength':
        cur_path = config.train_strength_path
    elif data_flag == 'degree':
        cur_path = config.train_degree_path
    else:
        print("输入的data_flag错误！")
        return
    all_set = load_data(dir_path=cur_path)
    # 数据分割
    train_data, dev_data = split_dataset_pd(dataset=all_set, seed=2)
    # 数据集加载
    train_dataset = MyDataset(config=config, dataset=train_data)
    dev_dataset = MyDataset(config=config, dataset=dev_data)

    # 数据不均衡，重要性采样
    if data_flag == 'score':
        cur_class_name = config.score_every_class
    elif data_flag == 'scale':
        cur_class_name = config.scale_every_class
    elif data_flag == 'influence':
        cur_class_name = config.influence_every_class
    elif data_flag == 'strength':
        cur_class_name = config.strength_every_class
    elif data_flag == 'degree':
        cur_class_name = config.degree_every_class
    else:
        print("输入的data_flag错误！")
        return
    original_weights = 1.0 / torch.tensor(cur_class_name, dtype=torch.float)
    train_targets = train_dataset.get_all_classes()
    sample_weights = original_weights[train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 构建数据的dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False,
                                  sampler=sampler)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = ClassifierWithBert4Layer(config=config, flag=data_flag).to(config.device)
    mytrain(config=config, model=model, train_dataset=train_dataloader, dev_dataset=dev_dataloader, writer=writer)
    # 记录训练总时长
    time_end = time.time()
    print(f"总计训练耗时：{time_end - time_start} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Event clue model training.")
    parser.add_argument("--flag_id", default=1, type=int, help="Choose the data type for training(scope:[0,1,2,3,4]).")
    args = parser.parse_args()
    if args.flag_id not in [0, 1, 2, 3, 4]:
        print("选择了错误的数据flag！")
    else:
        flags = ['score', 'scale', 'influence', 'strength', 'degree']
        cur_id = args.flag_id
        main(data_flag=flags[cur_id])

    # python trainer.py --flag_id 2
