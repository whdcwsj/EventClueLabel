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
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import argparse

from model.model import ClassifierWithBert4Layer
from config.config import Config
from dataloader import load_data, split_dataset_pd, MyDataset


# 数据不平衡，以Micro-F1为基准进行评判
# 多分类问题中， micro-f1与accuracy存在恒等性
def mytrain(config, model, train_dataset, dev_dataset, writer, train_metrics):
    # (precision, recall, macro_f1, _), micro_f1, dev_accuracy, dev_loss = myevaluate(config=config, model=model,
    #                                                                                 dev_dataset=dev_dataset)
    # return
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    pre_best_performance = 0
    improved_epoch = 0
    # 记录最佳模型的效果
    best_record = {
        'epoch': -1,
        'macro_f1': -1,
        'micro_f1': -1,
        'accuracy': -1,
        'precision': -1,
        'recall': -1
    }
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

        (precision, recall, macro_f1, _), micro_f1, dev_accuracy, dev_loss = myevaluate(config=config, model=model,
                                                                                        dev_dataset=dev_dataset)
        writer.add_scalar(tag='loss/train', scalar_value=total_loss / len(train_dataset), global_step=epoch)
        writer.add_scalar(tag='loss/dev', scalar_value=dev_loss / len(dev_dataset), global_step=epoch)
        writer.add_scalars(main_tag='performance/f1', tag_scalar_dict={'micro_f1': micro_f1, 'macro_f1': macro_f1},
                           global_step=epoch)
        writer.add_scalar(tag='performance/accuracy', scalar_value=dev_accuracy, global_step=epoch)
        writer.add_scalar(tag='performance/precision', scalar_value=precision, global_step=epoch)
        writer.add_scalar(tag='performance/recall', scalar_value=recall, global_step=epoch)
        if train_metrics is 'macro':
            my_metrics = macro_f1
        else:
            my_metrics = micro_f1
        if pre_best_performance < my_metrics:
            pre_best_performance = macro_f1
            improved_epoch = epoch
            torch.save(model.state_dict(), config.save_model_path)
            best_record['epoch'] = epoch
            best_record['macro_f1'] = macro_f1
            best_record['micro_f1'] = micro_f1
            best_record['accuracy'] = dev_accuracy
            best_record['precision'] = precision
            best_record['recall'] = recall
            print("model saved!!!")
        elif epoch - improved_epoch >= config.require_improvement:
            print("model didn't improve for a long time!So break!!!")
            break
    writer.close()
    print("~最佳模型结果~")
    print(f"epoch: {best_record['epoch']}")
    print(f"macro_f1: {best_record['macro_f1']}\n")
    print(f"micro_f1: {best_record['micro_f1']}")
    print(f"accuracy: {best_record['accuracy']}\n")
    print(f"precision: {best_record['precision']}")
    print(f"recall: {best_record['recall']}")


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
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Macro-F1: {macro_scores[2]}\n")
    print(f"Micro-F1: {micro_scores[2]}\n")
    print("Classification Report \n", classification_report(y_true=y_true, y_pred=y_pred, digits=4))

    return macro_scores, micro_scores[2], accuracy, evaluate_loss


def main(data_flag, performance_metrics):
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
    mytrain(config=config, model=model, train_dataset=train_dataloader, dev_dataset=dev_dataloader, writer=writer,
            train_metrics=performance_metrics)
    # 记录训练总时长
    time_end = time.time()
    print(f"总计训练耗时：{time_end - time_start} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Event clue model training.")
    parser.add_argument("--flag_id", default=0, type=int, help="Choose the data type for training(scope:[0,1,2,3,4]).")
    parser.add_argument("--metrics", default='macro', type=str,
                        help="Choose the metrics for model training.(scope:micro,macro)")
    args = parser.parse_args()
    if args.flag_id not in [0, 1, 2, 3, 4]:
        print("选择了错误的数据flag！")
    else:
        flags = ['score', 'scale', 'influence', 'strength', 'degree']
        cur_id = args.flag_id
        cur_metrics = args.metrics
        if cur_id not in [0, 1, 2, 3, 4]:
            print("选择错误的训练数据集ID")
        elif cur_metrics not in ['micro', 'macro']:
            print("选择错误的训练衡量指标")
        else:
            main(data_flag=flags[cur_id], performance_metrics=cur_metrics)

    # python trainer.py --flag_id 2
