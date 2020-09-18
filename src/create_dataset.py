import ipdb
import sys
import argparse
import logging
import json
import os
import pickle
import torch
import jieba
from dataset import news_Dataset
from tqdm import tqdm
from random import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/training_data.pkl")
    parser.add_argument('--save_dataset_dir', type=str, default="./data/")
    parser.add_argument('--sorted_dict_dir', default="data/sorted_dict.pkl")
    parser.add_argument('--people_name_dir', default="data/my_dict.txt")
    parser.add_argument('--max_sentence_len', type=int, default=200, help='data max padding length')
    args = parser.parse_args()
    return args


class Preprocessor:
    def __init__(self, sorted_dict, people_name_dir, max_sentence_len):
        self.sorted_dict = sorted_dict
        jieba.load_userdict(people_name_dir)
        self.max_sentence_len = max_sentence_len
        self.tags_dict = {'business': 0, 'politics': 1, 'society': 2, 'world': 3, 'entertainment': 4}

    """
    Args:
        datas: list of dict
        
        data: {
            'content': 'chinese words',
            'tags': 'business'
        }
    """
    def create_dataset(self, datas):
        news_content = []
        unk = self.sorted_dict['<UNK>']
        shuffle(datas)

        for news in tqdm(datas, leave=False, position=0, dynamic_ncols=True):
            cut_news = jieba.cut(news['content'])
            cut_news = [e for e in cut_news]

            # Remove trivial news head.  comment shuffle and set news = datas[60004] to understand why
            if '〕' in cut_news:
                cut_news = cut_news[cut_news.index('〕') + 1:]
            if '\r' in cut_news:
                cut_news = cut_news[:cut_news.index('\r')]
            while ' ' in cut_news:
                cut_news.remove(' ')
            while '\xa0' in cut_news:
                cut_news.remove('\xa0')
            while '\u3000' in cut_news:
                cut_news.remove('\u3000')

            # Trim content to final period and add padding.
            stop = min(self.max_sentence_len - 2, len(cut_news))  # -2 for <BOS> & <EOS>
            while stop > 0 and cut_news[stop - 1] not in ['。', '！', '」']:
                stop -= 1
            cut_news = ['<BOS>'] + cut_news[:stop] + ['<EOS>']
            cut_news = cut_news + ['<PAD>'] * ((self.max_sentence_len + 1) - len(cut_news))

            # convert content into indices and save it
            news_content.append([self.sorted_dict.get(word, unk) for word in cut_news])

        # (56032, 200)
        news_tags = torch.stack(
            [torch.LongTensor(self.max_sentence_len).fill_(self.tags_dict[data['tags']]) for data in datas])

        training_size = len(news_content) // 10 * 8
        train_data = {
            'news_content': news_content[: training_size],
            'news_tags': news_tags[: training_size]
        }
        valid_data = {
            'news_content': news_content[training_size:],
            'news_tags': news_tags[training_size:]
        }

        return news_Dataset(train_data), news_Dataset(valid_data)


def main(args):
    print('Loading pickles.')
    with open(args.data_dir, 'rb') as f:
        datas = pickle.load(f)
    with open(args.sorted_dict_dir, 'rb') as f:
        sorted_dict = pickle.load(f)

    preprocessor = Preprocessor(sorted_dict, args.people_name_dir, args.max_sentence_len)

    logging.info('Processing dataset.')
    train_dataset, valid_dataset = preprocessor.create_dataset(datas)

    logging.info('Saving processed valid dataset.')
    with open(args.save_dataset_dir + 'processed_valid_dataset_full_200.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    logging.info('Saving processed training dataset.')
    with open(args.save_dataset_dir + 'processed_train_dataset_full_200.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhood = ipdb.set_trace
        args = parse_args()
        main(args)

