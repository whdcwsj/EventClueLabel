# encoding: utf-8
# @author: 王宇静轩
# @file: file_compare.py
# @time: 2022/11/5 下午8:05

# 老李头作业
# 通过ifidf比较两个文档的的摘要或者全文的文本相似度
# 使用sklearn中的TfidfVectorizer方法来进行


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
#
# vectorizer = TfidfVectorizer()
# # fit_transform:学习词汇和idf，返回文档术语矩阵。
# X = vectorizer.fit_transform(corpus)
# # 每个词的词向量
# print(X)
# # get_feature_names:从特征整数索引到特征名称的数组映射。
# print(vectorizer.get_feature_names())
# print(X.shape)
#
# print(X.toarray())


import os
os.environ['PYTHONHASHSEED'] = '0'

import sys
import numpy as np
import hashlib



# 测试代码
d = 10  # d是维度
k = 3  # k是k-gram，滑动窗口数值

# 0824c77ecd7376ee95d34be39f5ef404   9
# 0824c77ecd7376ee95d34be39f5ef404   0

def main():
    with open("华夏.txt", "r", encoding='UTF-8') as f:
        text = f.read()

    freq = [0 for i in range(d)]  # 创建长度为d的列表，初始值0
    print(freq)
    print(len(text))
    for i in range(len(text) - k):  # 循环抽取k- grams,计算频率
        kgram = text[i:i + k]
        print(kgram)
        # hash() 用于获取取一个对象（字符串或者数值等）的哈希值
        # 内置hash函数带有随机magic的功能，不能保证唯一性
        # 因此采用hashlib的md5码
        # temp_key = hashlib.md5(kgram.encode(encoding='UTF-8')).hexdigest()
        temp_key = hash(kgram)
        print(temp_key)
        print(temp_key % d)
        freq[temp_key % d] += 1
    print(freq)
    print(type(freq))
    vector = np.array(freq)  # 创建文档摘要向量
    print(vector)
    print(type(vector))


if __name__ == '__main__':
    main()
