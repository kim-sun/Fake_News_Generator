import torch
import pickle
import jieba
from collections import Counter
from os import listdir
import gc
from tqdm import tqdm


# cut sentence with jieba
words = []

jieba.load_userdict("my_dict.txt")

path = "./data/"
files = listdir(path)
for file in files:
    with open(path + file, 'rb') as f:
        data = pickle.load(f)
        print("loading ", file, ":", len(data))
        for news in data:
            cutted_word = jieba.cut(news['content'])
            for word in cutted_word:
                words.append(word)

print('load all the words: ', len(words))

# count the words
processed_words = {}
words_counter = Counter(words)
for word in words_counter:
    if words_counter[word] > 7:
        processed_words[word] = 1

print("common words: ", len(processed_words))

#save
with open('counter.pkl', 'wb') as f:
    pickle.dump(words_counter, f)
with open('words.pkl', 'wb') as f:
    pickle.dump(processed_words, f)

print("save counter & words")

#delete variables
del words
del words_counter
gc.collect()

# check fasttext
import re

vectors = []
word_dict = {}

with open('cc.zh.300.vec', encoding = 'utf8') as fp:

    row1 = fp.readline()
    # if the first row is not header
    if not re.match('^[0-9]+ [0-9]+$', row1):
        # seek to 0
        fp.seek(0)
    # otherwise ignore the header

    for i, line in tqdm(enumerate(fp), desc="fasttext"):
        cols = line.rstrip().split(' ')
        word = cols[0]
        # print(word)
        # skip word not in words if words are provided
        if processed_words.get(word) is None:
            continue
        elif word_dict.get(word) is None:
            word_dict[word] = len(word_dict)
            vectors.append([float(v) for v in cols[1:]])

with open('word_dic.pkl', 'wb') as f:
    pickle.dump(word_dict, f)
with open('vector.pkl', 'wb') as f:
    pickle.dump(vectors, f)

print("save dic & vectors")
