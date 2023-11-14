import numpy as np
import pandas as pd
import torch
import jieba
import re


stop_words = pd.read_csv("stop_words.txt" , header = None , names = ["text"])

# print(type(stop_words))

def chinese_pre(text_data):
    text_data = text_data.lower()
    text_data = re.sub("\d+", "", text_data)
    a = jieba.cut(text_data, cut_all=True)
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    text_data = " ".join(a)
    return text_data


text_data = "做工真的不错，薄！散热好，几乎听不到噪音！性价比最高了，拿来日常办公的确够用，但是游戏发烧友就不要考虑啦！毕竟是集显，跑不动游戏，总而言之，作为学生上课，处理办公文件，看个剧，可以直接冲"

# td = file.open()

text_data = chinese_pre(text_data).split(" ")

# print(text_data)


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

word2id_freq, word2id_dict, id2word_dict = build_dict(text_data)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))
