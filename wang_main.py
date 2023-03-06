# from transformers import BertTokenizer
# import torch
#
# vocab_path = './data/bert_chinese/vocab.txt'
# tokenizer = BertTokenizer(vocab_path)
# s_a, s_b = "李白拿了个锤子", "锤子？"
# input_id = tokenizer(s_a, max_length=20 ,padding=True)
# print(input_id)
# input_id = torch.tensor([input_id])  # 输入数据是tensor且batch形式的
# print(input_id)


import numpy as np
import torch

# score_label_map = {
#     -5: 0,
#     -3: 1,
#     -1: 2,
#     1: 3,
#     3: 4,
#     5: 5,
# }
#
# wang = [5, 3, -1]
# wang = np.array(wang)
# print(wang)
# print(type(wang))

# si = [score_label_map[i] for i in wang]
# print(si)
# print(type(si))

# original_weights = 1.0 / torch.tensor([1, 2, 3], dtype=torch.float)
# map_train_targets = np.array([0, 2, 1])
#
# sample_weights = original_weights[map_train_targets]
# print(sample_weights)


# wang = [1, 2, 3]
# si = [4, 5, 6]
# wang += si
# print(wang)


# import torch
#
#
# def device_distribute(input_model_count):
#     model_gpu_correspond = {}
#     model_count = input_model_count
#     gpu_count = torch.cuda.device_count()
#     gpu_ids = [int(_) for _ in range(gpu_count)]
#     gpu_devices = ["cuda:{}".format(_) for _ in gpu_ids if torch.cuda.is_available()]
#
#     interval = model_count / gpu_count
#     if gpu_count > 0:
#         if interval <= 1:
#             for i in range(model_count):
#                 model_gpu_correspond[i] = gpu_devices[i]
#         else:
#             cur_gpu = 0
#             cur_interval = interval
#             for i in range(model_count):
#                 if cur_interval >= (i + 1):
#                     model_gpu_correspond[i] = gpu_devices[cur_gpu]
#                 else:
#                     cur_interval += interval
#                     cur_gpu += 1
#                     model_gpu_correspond[i] = gpu_devices[cur_gpu]
#
#     else:
#         for i in range(model_count):
#             model_gpu_correspond[i] = "cpu"
#
#     return model_gpu_correspond
#
#
# res = device_distribute(input_model_count=5)
# print(res)


# gpu_count = 4
# # model_count = 3
# model_count = 6
#
# if interval <= 1:
#     for i in range(model_count):
#         model_gpu_correspond[i] = gpu_devices[i]
# else:
#     cur_gpu = 0
#     cur_interval = interval
#     for i in range(model_count):
#         if cur_interval >= (i + 1):
#             model_gpu_correspond[i] = gpu_devices[cur_gpu]
#         else:
#             cur_interval += interval
#             cur_gpu += 1
#             model_gpu_correspond[i] = gpu_devices[cur_gpu]
#
# print(model_gpu_correspond)

# gpu_device = torch.cuda.is_available()
# gpu_count = torch.cuda.device_count()
# gpu_name = torch.cuda.get_device_name(0)
# gpu_cur = torch.cuda.current_device()
#
# print(gpu_device)
# print(gpu_count)
# print(gpu_name)
# print(gpu_cur)



weight_dict = {
    0: "10_21_data_event_score10-22_18.00.ckpt",
    1: "10_21_data_event_scale10-22_18.00.ckpt",
    2: "10_21_data_event_influence10-22_18.00.ckpt",
    3: "10_21_data_event_strength10-23_10.01.ckpt",
    4: "10_21_data_event_degree10-23_10.01.ckpt"
}

print(type(weight_dict[0]))
print(666)
