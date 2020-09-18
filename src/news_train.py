import torch
from torch.utils.data import DataLoader

import argparse
import traceback
import os
import sys
import ipdb
import pickle
import numpy as np
from tqdm import tqdm
from collections import deque
from matplotlib import pyplot as plt
from news_net import News_Net


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="./data/")
    parser.add_argument('--sorted_dict', default="data/sorted_dict.pkl")
    parser.add_argument('--sorted_vector', default="data/sorted_vector.pkl")
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tag_types', type=int, default=5)
    parser.add_argument('--save_model_path', type=str, default='./model/')
    parser.add_argument('--plt_freq', type=int, default=50)

    args = parser.parse_args()
    return args


def save(model, optimizer, path):
    print('Saving model')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)


def load(model, optimizer, path):  # optimizer
    model.load_state_dict(torch.load(path)['model'])
    optimizer.load_state_dict(torch.load(path)['optimizer'])


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    print('Loading dataset.')
    with open(args.dataset + 'processed_train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(args.dataset + 'processed_valid_dataset.pkl', 'rb') as f:
        valid_dataset = pickle.load(f)
    with open(args.sorted_dict, 'rb') as f:
        sorted_dict = pickle.load(f)
    with open(args.sorted_vector, 'rb') as f:
        sorted_vector = pickle.load(f)

    voc_size = len(sorted_dict)
    emb_size = len(sorted_vector[0])
    print('voc_size: %d \ntraining size: %d\nvalid size: %d\n' % (voc_size, len(train_dataset), len(valid_dataset)))

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)
    model = News_Net(voc_size, emb_size, args.tag_types).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    plt_epoch, plt_train_loss, plt_valid_loss = [], [], []
    # avg_loss_plt = deque(maxlen=args.plt_freq)
    train_avg_loss, valid_avg_loss = [], []

    print('Start training.')
    for epoch in range(args.epoch):
        plt_epoch.append(epoch + 1)

        lr = args.lr * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # training
        trange = tqdm(train_dataloader, total=len(train_dataloader), desc='training')
        model.train()
        for batch in trange:
            train = batch['train'].to(device)       # (batch, 200)
            label = batch['label'].to(device)       # (batch, 200)
            tags = batch['tags'].to(device)         # (batch, 5)
            content_loss, tag_loss = model.forward(train, label, tags)

            loss = content_loss + tag_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_avg_loss.append(loss.item())
            trange.set_postfix(epoch='%d/%d' % (epoch+1, args.epoch),
                               c_loss=content_loss.item(), t_loss=tag_loss.item())
        plt_train_loss.append(np.mean(train_avg_loss))
        train_avg_loss = []

        # validating
        trange = tqdm(valid_dataloader, total=len(valid_dataloader), desc='validating')
        model.eval()
        for batch in trange:
            train = batch['train'].to(device)  # (batch, 200)
            label = batch['label'].to(device)  # (batch, 200)
            tags = batch['tags'].to(device)  # (batch, 5)

            with torch.no_grad():
                content_loss, tag_loss = model.forward(train, label, tags)

            loss = content_loss + tag_loss
            valid_avg_loss.append(loss.item())
            trange.set_postfix(epoch='%d/%d' % (epoch + 1, args.epoch),
                               c_loss=content_loss.item(), t_loss=tag_loss.item())
        plt_valid_loss.append(np.mean(valid_avg_loss))
        valid_avg_loss = []

        plot_figure(plt_epoch, plt_train_loss, plt_valid_loss, args.save_model_path)
        save(model, optimizer, os.path.join(args.save_model_path, 'epoch_{}.ckpt'.format(epoch+1)))


def plot_figure(plt_iter, train_loss, valid_loss, save_path):
    plt.clf()
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(plt_iter, train_loss, label='training')
    plt.plot(plt_iter, valid_loss, '--', label='evaluating')
    plt.legend(loc='best')
    plt.grid(True)
    save_path += 'Learning_Curve.png'

    print('Saving learning curve.')
    plt.savefig(save_path)


if __name__ == "__main__":
    try:
        args = parse()
        news = main(args)

    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
