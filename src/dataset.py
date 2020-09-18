import torch
from torch.utils.data import Dataset
import ipdb


class news_Dataset(Dataset):
    """
    Args:
        data (dict): {
            'news_content': []
            'news_tags': torch.LongTensor
        }

        data['news_content']: size:(56032, 201)
            [[2, 35, 6, 1577, 230, ...], [2, 35, 6, 1577, 230, ...], ...]

        data['news_tags']: size:(56032, 200)
            tensor[(1,1,1,1,1,...), (3,3,3,3,3,...), ...]
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['news_content'])

    def __getitem__(self, index):
        data = {
            'news_content': self.data['news_content'][index],
            'news_tags': self.data['news_tags'][index]
        }
        return data

    def collate_fn(self, datas):

        batch = {}

        news_content = [data['news_content'] for data in datas]     # list of [2, 35, 6, 1577, 230, ...]
        news_tags = [data['news_tags'] for data in datas]           # list of tensor(1,1,1,1,1,...)

        batch['train'] = torch.LongTensor([news[:-1] for news in news_content])  # (batch, 200)
        batch['label'] = torch.LongTensor([news[1:] for news in news_content])   # (batch, 200)
        batch['tags'] = torch.stack(news_tags)                                   # (batch, 200)
        return batch
