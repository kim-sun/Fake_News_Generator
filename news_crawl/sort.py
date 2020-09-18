import pickle
from collections import Counter
import torch
import pdb
from tqdm import tqdm

with open('counter.pkl', 'rb') as f:
    counter = pickle.load(f)
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('word_dic.pkl', 'rb') as f:
    word_dict = pickle.load(f)

with open('vector.pkl', 'rb') as f:
    vectors = pickle.load(f)

print(len(words))
print(len(counter))
print(len(word_dict))
print(len(vectors))

count_words = counter.most_common(88452)
for i, word in enumerate(count_words):
    if i > 10:
        break
    print(word)

new_word_dict = {'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<UNK>': 3}
new_vectors = torch.nn.init.uniform_(torch.empty(4, 300))

for word in tqdm(count_words, desc="sorted list of vector"):
    index = word_dict.get(word[0])
    if index is not None:
        v = torch.tensor(vectors[index]).unsqueeze(0)
    else:
        v = torch.nn.init.uniform_(torch.empty(1, 300))

    new_vectors = torch.cat([new_vectors, v], 0)
    new_word_dict[word[0]] = len(new_word_dict)

with open('sorted_dict.pkl', 'wb') as f:
    pickle.dump(new_word_dict, f)

with open('sorted_vector.pkl', 'wb') as f:
    pickle.dump(new_vectors, f)




