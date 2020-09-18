# coding=gbk

import torch
import argparse
import traceback
import os
import sys
import ipdb
import jieba
import pickle
from news_net import News_Net


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sorted_dict', default="data/sorted_dict.pkl")
    parser.add_argument('--sorted_vector', default="data/sorted_vector.pkl")
    parser.add_argument('--people_name', default="data/my_dict.txt")
    parser.add_argument('--model_path', default="model/epoch_12.ckpt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--p_len', type=int, default=200)
    parser.add_argument('--given', type=str, default="")
    parser.add_argument('--tag', type=str, default="politics",
                        choices=['business', 'politics', 'society', 'world', 'entertainment'])

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fake_news_path = 'fake_news/'
    if not os.path.exists(fake_news_path):
        os.makedirs(fake_news_path)

    with open(args.sorted_dict, 'rb') as file_2:
        sorted_dict = pickle.load(file_2)
    with open(args.sorted_vector, 'rb') as file_3:
        sorted_vector = pickle.load(file_3)

    tags_dict = {'business': 0, 'politics': 1, 'society': 2, 'world': 3, 'entertainment': 4}
    tag = tags_dict[args.tag]

    voc_size = len(sorted_dict)
    emb_size = len(sorted_vector[0])
    model = News_Net(voc_size, emb_size, tage_types=len(tags_dict)).to(device)
    print('load model from', args.model_path)
    model.load_state_dict(torch.load(args.model_path)['model'])

    generator(args.given, model, sorted_dict, args.people_name, device, args.p_len, tag)


def generator(given, model, sorted_dict, people_name, device, p_len, tag):

    jieba.load_userdict(people_name)
    given = jieba.cut(given)
    result = ['<BOS>'] + [e for e in given]

    unk = sorted_dict['<UNK>']

    skip_word_index = [sorted_dict[skip_word] for skip_word in ['<PAD>', '<BOS>', '<UNK>']]  # '<EOS>'
    given_indices = torch.LongTensor([[sorted_dict.get(word, unk) for word in result]]).to(device)

    new_dict = {v: k for k, v in sorted_dict.items()}

    while True:
        tag_vector = torch.LongTensor(given_indices.shape).fill_(tag).to(device)
        pred_index = model.predict(given_indices, tag_vector, skip_word_index)
        pred_word = new_dict[pred_index.item()]
        result.append(pred_word)
        if len(result) % 50 == 0:
            print(''.join(result) + '\n')
        # if pred_word in ['。', '！'] and len(result) > p_len:
        if pred_word == '<EOS>':
            break
        else:
            given_indices = torch.cat((given_indices, pred_index.unsqueeze(0)), 1)

    fake_news = ''.join(result[1:-1])
    print(fake_news)


if __name__ == "__main__":
    try:
        args = parse()
        main(args)

    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
