import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
##输出图显示中文
from matplotlib.font_manager import FontProperties

fonts = FontProperties(fname=" /Library/Fonts/华文细黑.ttf")
import re
import string
import copy
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
import torchtext
# from torchtext.legacy import data
#from torchtext.vocab import vectors
#from torchtext.legacy.data import Field, TabularDataset, Iterator, BucketIterator
stop_words = pd.read_csv("stop_words.txt", header=None, names=["text"])


def chinese_pre(text_data):
    # 操字母转化为小写，丢除数字，
    text_data = text_data.lower()
    text_data = re.sub("\d+", "", text_data)
    ##分词,使用精确模式
    a = jieba.cut(text_data, cut_all=True)
    # 毋丢停用词和多余空格
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    # 贽处理后的词语使用空格连接为字符串
    text_data = " ".join(a)
    # print(text_data)
    return text_data

class DataProcessor(object):
    


    def read_text(self, is_train_data):
        if (is_train_data):
            train_df = pd.read_excel("cnews_val.et", header=None, names=["label", "text"])

            ##对中文文本数据进行预处理，去除一些不需要的字符，分词，去停用词，等操作
            train_df["cutword"] = train_df.text.apply(chinese_pre)
            train_df[["label", "cutword"]].to_csv("cnews_train.csv", index=False)
            return train_df["cutword"], train_df["label"]

    def word_count(self, datas):
        # 统计单词出现的频次，并将其降序排列，得出出现频次最多的单词
        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                if (word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item: item[1], reverse=True)
        return word_count_sorted

    def word_index(self, datas, vocab_size):
        # 创建词表
        word_count_sorted = self.word_count(datas)
        word2index = {}
        # 词表中未出现的词
        word2index["<unk>"] = 0
        # 句子添加的padding
        word2index["<pad>"] = 1

        # 词表的实际大小由词的数量和限定大小决定
        vocab_size = min(len(word_count_sorted), vocab_size)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + 2
        return word2index, vocab_size

    def get_datasets(self, vocab_size, embedding_size, max_len):
        # 注，由于nn.Embedding每次生成的词嵌入不固定，因此此处同时获取训练数据的词嵌入和测试数据的词嵌入
        # 测试数据的词表也用训练数据创建
        train_datas, train_labels = self.read_text(is_train_data=True)
        word2index, vocab_size = self.word_index(train_datas, vocab_size)

        train_features = []
        for data in train_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower()  # 词表中的单词均为小写
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])  # 词表中未出现的词用<unk>代替
                if (len(feature) == max_len):  # 限制句子的最大长度，超出部分直接截断
                    break
            # 对未达到最大长度的句子添加padding
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            train_features.append(feature)

        train_features = torch.LongTensor(train_features)
        train_labels = torch.LongTensor(train_labels)

        # 将词转化为embedding
        # 词表中有两个特殊的词<unk>和<pad>，所以词表实际大小为vocab_size + 2
        embed = nn.Embedding(vocab_size + 2, embedding_size)
        train_features = embed(train_features)
        # 指定输入特征是否需要计算梯度

        train_datasets = torch.utils.data.TensorDataset(train_features, train_labels)

        return train_datasets