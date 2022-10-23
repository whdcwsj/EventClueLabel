import json
import csv
import pandas as pd
import os

score_label_map = {
    -5: 1,
    -3: 2,
    -1: 3,
    1: 4,
    3: 5,
    5: 6,
}


def construct_event_score():
    # 3229条数据
    filename = '2022-10-13_event.json'
    out_file = 'event_score.csv'

    csv_file = open(out_file, 'w', newline='')
    writer = csv.writer(csv_file)

    header_label = 'event,score'
    csv_file.write(header_label + '\n')

    with open(filename, 'r') as f:
        # 返回列表
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # str转dict
            line_text = json.loads(line)
            if line_text['source']['score'] is not None and line_text['source']['eventName'] is not None:
                cur_list = []
                cur_event = line_text['source']['eventName'].strip()
                cur_score = line_text['source']['score']
                cur_list.append(cur_event)
                cur_list.append(cur_score)
                writer.writerow(cur_list)


# 1、事件打分（6分类）
def construct_event_score_1_3_5():
    # 1985条数据
    filename = '2022-10-21_event.json'
    out_file = 'event_score_1_3_5.csv'

    if not os.path.exists(out_file):
        print("当前输出文件不存在")
        return

    csv_file = open(out_file, 'w', newline='')
    writer = csv.writer(csv_file)

    header_label = 'event,score'
    csv_file.write(header_label + '\n')

    with open(filename, 'r') as f:
        # 返回列表
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # str转dict
            line_text = json.loads(line)
            if line_text['source']['score'] is not "" and line_text['source']['eventName'] is not "":
                if int(float(line_text['source']['score'])) in [-5, -3, -1, 1, 3, 5]:
                    cur_list = []
                    cur_event = line_text['source']['eventName'].strip()
                    cur_score = score_label_map[int(float(line_text['source']['score']))]
                    cur_list.append(cur_event)
                    cur_list.append(cur_score)
                    writer.writerow(cur_list)


# 2、事件规模（3分类）
def construct_event_scale_military():
    filename = '2022-10-21_event.json'
    out_file = 'event_scale.csv'

    if not os.path.exists(out_file):
        print("当前输出文件不存在")
        return

    csv_file = open(out_file, 'w', newline='')
    writer = csv.writer(csv_file)

    header_label = 'event,scale'
    csv_file.write(header_label + '\n')

    with open(filename, 'r') as f:
        # 返回列表
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # str转dict
            line_text = json.loads(line)
            if line_text['source']['primaryClassification'] is not "" and line_text['source']['scale'] \
                    is not "" and line_text['source']['eventName'] is not "":
                if int(line_text['source']['primaryClassification']) == 2 and int(line_text['source']['scale']) != -1:
                    cur_list = []
                    cur_event = line_text['source']['eventName'].strip()
                    cur_scale = int(line_text['source']['scale'])
                    cur_list.append(cur_event)
                    cur_list.append(cur_scale)
                    writer.writerow(cur_list)


# 3、事件影响（2分类，正向负向）
def construct_event_influence_strategy():
    filename = '2022-10-21_event.json'
    out_file = 'event_influence.csv'

    if not os.path.exists(out_file):
        print("当前输出文件不存在")
        return

    csv_file = open(out_file, 'w', newline='')
    writer = csv.writer(csv_file)

    header_label = 'event,influence'
    csv_file.write(header_label + '\n')

    with open(filename, 'r') as f:
        # 返回列表
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # str转dict
            line_text = json.loads(line)
            if line_text['source']['primaryClassification'] is not "" and line_text['source']['scale'] \
                    is not "" and line_text['source']['eventName'] is not "":
                if int(line_text['source']['primaryClassification']) is not 2 and int(line_text['source']['scale']) \
                        != 3:
                    cur_list = []
                    cur_event = line_text['source']['eventName'].strip()
                    cur_scale = int(line_text['source']['scale'])
                    cur_list.append(cur_event)
                    cur_list.append(cur_scale)
                    writer.writerow(cur_list)


# 4、事件强度（3分类）
def construct_event_strength_military():
    filename = '2022-10-21_event.json'
    out_file = 'event_strength.csv'

    if not os.path.exists(out_file):
        print("当前输出文件不存在")
        return

    csv_file = open(out_file, 'w', newline='')
    writer = csv.writer(csv_file)

    header_label = 'event,strength'
    csv_file.write(header_label + '\n')

    with open(filename, 'r') as f:
        # 返回列表
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # str转dict
            line_text = json.loads(line)
            if line_text['source']['primaryClassification'] is not "" and line_text['source']['strength'] \
                    is not "" and line_text['source']['eventName'] is not "":
                if int(line_text['source']['primaryClassification']) == 2 and int(line_text['source']['strength']) != 4:
                    cur_list = []
                    cur_event = line_text['source']['eventName'].strip()
                    cur_scale = int(line_text['source']['strength'])
                    cur_list.append(cur_event)
                    cur_list.append(cur_scale)
                    writer.writerow(cur_list)


# 5、事件程度（4分类）
def construct_event_degree_strategy():
    # 1970
    filename = '2022-10-21_event.json'
    out_file = 'event_degree.csv'

    if not os.path.exists(out_file):
        print("当前输出文件不存在")
        return

    csv_file = open(out_file, 'w', newline='')
    writer = csv.writer(csv_file)

    header_label = 'event,degree'
    csv_file.write(header_label + '\n')

    with open(filename, 'r') as f:
        # 返回列表
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # str转dict
            line_text = json.loads(line)
            if line_text['source']['primaryClassification'] is not "" and line_text['source']['strength'] \
                    is not "" and line_text['source']['eventName'] is not "":
                if int(line_text['source']['primaryClassification']) is not 2 and int(line_text['source']['strength']) \
                        in [1, 2, 3, 4]:
                    cur_list = []
                    cur_event = line_text['source']['eventName'].strip()
                    cur_scale = int(line_text['source']['strength'])
                    cur_list.append(cur_event)
                    cur_list.append(cur_scale)
                    writer.writerow(cur_list)


# 读取csv文件，去除重复数据，统计每个lable下的数据量
def read_csv_file_record_and_remove_repeated(filename, label):
    df = pd.read_csv(filename, sep=',')
    print(f"当前数据总量：{len(df)}")
    # keep:first 保存第一次出现的重复项
    df.drop_duplicates(subset=['event'], keep='first', inplace=True)
    df.to_csv(filename, index=False)

    print(f"去重后数据总量：{len(df)}")
    print(df.keys())
    # print(df['score'])

    count = df.groupby([label], as_index=False)[label].agg({'cnt': 'count'})
    print(count)
    # print(df[df['score'] == 9.0])


if __name__ == "__main__":
    # construct_event_score()
    # print("提取事件打分数据完成")

    # construct_event_score_1_3_5()
    # print("提取1-3-5事件打分数据完成")

    # construct_event_scale_military()
    # print("提取事件规模数据完成(J事)")

    # construct_event_influence_strategy()
    # print("提取事件影响数据完成(战略)")

    construct_event_strength_military()
    print("提取事件强度数据完成(J事)")

    # construct_event_degree_strategy()
    # print("提取事件强度数据完成(战略)")

    filename = 'event_strength.csv'
    label = 'strength'
    read_csv_file_record_and_remove_repeated(filename=filename, label=label)
    print("不同标签的数据量统计完成")
