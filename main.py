import numpy as np
import pandas as pd
import torch
import jieba
import re
import torchtext
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 停用词
stop_words = pd.read_csv("stop_words.txt" , header = None , names = ["text"])
stop_words = stop_words["text"].tolist()

# 中文预处理函数，去除其中的停用词，并进行分词
def chinese_pre(text_data):
    text_data = text_data.lower()
    cutted_text_data_list = jieba.cut(text_data)
    filtered_word = [word for word in cutted_text_data_list if word not in stop_words]

    # text_data = [word for word in text_data if word not in stop_words]
    text_data = " ".join(filtered_word)
    return text_data


# 输入正向评论

pos_file = open("data_pos.txt")
pos_lines = pos_file.readlines()

# 输入负向评论

neg_file = open("data_neg.txt")
neg_lines = neg_file.readlines()

# 预处理

pos = []
neg = []

for line in pos_lines:
    line = chinese_pre(line).split(" ")

    for word in line:
        if (word != "\n" and word != ""):
            pos.append(word)

for line in neg_lines:
    line = chinese_pre(line).split(" ")

    for word in line:
        if (word != "\n" and word != ""):
            neg.append(word)

# testing pos & neg

# print("positive comments:")
# print(pos)

# print("negative comments:")
# print(neg)

'''
def build_dict(corpus):
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)

    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict

word2id_freq, word2id_dict, id2word_dict = build_dict(pos)

vocab_size = len(word2id_freq)


print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))
'''

# print(pos)


all_lines = pos + neg

# 贴标签
labels = [1] * len(pos) + [0] * len(neg)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_lines, labels, test_size=0.2, random_state=42)

# 使用TF-IDF转换器
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 训练SVM模型
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# 预测测试集
y_pred = svm_model.predict(X_test_tfidf)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

# 获取特征名（词）
feature_names = vectorizer.get_feature_names_out()

# 获取模型的系数
coefficients = svm_model.coef_.toarray()[0]

# print(coefficients)

# 将特征名和对应的系数绑定在一起
feature_weights = dict(zip(feature_names, coefficients))

# 对权重进行排序，输出最有影响的词
sorted_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)

# 输出排名前10的特征及其权重
print(sorted_features[:10])

# 对权重进行排序，输出对负面评论分类影响最大的词
sorted_negative_features = sorted(feature_weights.items(), key=lambda x: x[1])

# 输出排名前10的负面特征及其权重
print(sorted_negative_features[:10])