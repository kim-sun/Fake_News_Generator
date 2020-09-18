import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ipdb
import random
# from attention import Attention


class News_Net(nn.Module):
    def __init__(self, voc_size, emb_size, tage_types, projection_size=512):
        super(News_Net, self).__init__()
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.tage_types = tage_types
        self.projection_size = projection_size
        self.h_size = projection_size * 4

        self.embedding = nn.Embedding(voc_size, emb_size)
        self.tag_embedding = nn.Embedding(tage_types, emb_size)
        self.linear_in = nn.Linear(emb_size * 2, projection_size)

        self.rnn_1 = nn.LSTM(input_size=projection_size, hidden_size=self.h_size, batch_first=True)
        self.linear_1 = nn.Linear(self.h_size, projection_size)

        self.rnn_2 = nn.LSTM(input_size=projection_size, hidden_size=self.h_size, batch_first=True)
        self.linear_2 = nn.Linear(self.h_size, projection_size)
        self.dropout = torch.nn.Dropout(0.4)
        # self.rnn_3 = nn.LSTM(input_size=projection_size, hidden_size=self.h_size, batch_first=True)
        # self.linear_3 = nn.Linear(self.h_size, projection_size)

        self.adaptive_loss = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=projection_size, n_classes=voc_size, cutoffs=[100, 1000, 10000])

        self.linear_categories = nn.Linear(projection_size, tage_types)
        self.categories_loss = torch.nn.CrossEntropyLoss()

        # self.attention = Attention(hidden_emb=self.h_size)

    def forward(self, content, label, tags):
        """
        Args:
            content: (batch, p_len)
            label:   (batch, p_len)
            tags:    (batch, p_len)
        """

        x = self.embedding(content)     # (batch, p_len, emb_size)
        x_ = self.tag_embedding(tags)   # (batch, p_len, emb_size)
        x = torch.cat((x, x_), 2)       # (batch, p_len, emb_size * 2)

        x = self.linear_in(x)           # (batch, p_len, prj_size)
        x = self.dropout(x)             # add dropout
        x, _ = self.rnn_1(x, None)      # (batch, p_len, hid_size)
        x = self.dropout(x)             # add dropout
        x = self.linear_1(x)            # (batch, p_len, prj_size)
        x = self.dropout(x)             # add dropout
        x, _ = self.rnn_2(x, None)      # (batch, p_len, hid_size)
        x = self.dropout(x)             # add dropout
        x = self.linear_2(x)            # (batch, p_len, prj_size)
        # x, _ = self.rnn_3(x, None)      # (batch, p_len, hid_size)
        # x = self.linear_3(x)            # (batch, p_len, prj_size)

        loss = self.adaptive_loss(x.view(-1, self.projection_size), label.view(-1)).loss

        x_categ = self.linear_categories(x)  # (batch, p_len, 5)
        categ_loss = self.categories_loss(x_categ.view(-1, self.tage_types), tags.view(-1))

        return loss, categ_loss

    def predict(self, given, tags, skip_word_indices):
        """
        Args:
            given: (1, sent_len)
            tags:  (1, sent_len)
            skip_word_indices: [indices of special tokens: BOS, PAD, UNK]
        """

        x = self.embedding(given)       # (batch, p_len, emb_size)  # (1,2,300)
        x_ = self.tag_embedding(tags)  # (batch, p_len, emb_size)
        x = torch.cat((x, x_), 2)  # (batch, p_len, emb_size * 2)

        x = self.linear_in(x)           # (batch, p_len, prj_size)
        x, _ = self.rnn_1(x, None)      # (batch, p_len, hid_size)
        x = self.linear_1(x)            # (batch, p_len, prj_size)
        x, _ = self.rnn_2(x, None)      # (batch, p_len, hid_size)
        x = self.linear_2(x)            # (batch, p_len, prj_size)
        # x, _ = self.rnn_3(x, None)      # (batch, p_len, hid_size)
        # x = self.linear_3(x)            # (batch, p_len, prj_size)  # (1,2,512)

        log_prob = self.adaptive_loss.log_prob(x[0][-1].unsqueeze(0))
        log_prob, pred_index = torch.sort(log_prob, descending=True)  # tensor([[3, 12, 6, ..., 1069]])

        top_log_prob = F.softmax(log_prob[0][:10], dim=0)  # tensor([0.3039, 0.2163, 0.2156, 0.1405, 0.1237])
        action_distributions = Categorical(top_log_prob)

        count = 0
        while True:
            i = action_distributions.sample().item() if count < 100 else random.randint(0, 3)
            if not pred_index[0][i].item() in skip_word_indices:
                count = 0
                return pred_index[0][i: i+1]
            count += 1




