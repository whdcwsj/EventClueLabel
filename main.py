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

score_label_map = {
    -5: 0,
    -3: 1,
    -1: 2,
    1: 3,
    3: 4,
    5: 5,
}

wang = [5, 3, -1]
wang = np.array(wang)
# print(wang)
# print(type(wang))

si = [score_label_map[i] for i in wang]
print(si)
print(type(si))

# original_weights = 1.0 / torch.tensor([1, 2, 3], dtype=torch.float)
# map_train_targets = np.array([0, 2, 1])
#
# sample_weights = original_weights[map_train_targets]
# print(sample_weights)


# wang = [1, 2, 3]
# si = [4, 5, 6]
# wang += si
# print(wang)
