import torch
import torch.nn as nn
import ipdb


class Attention(nn.Module):
    """
    Args:
        seq_len: maximum sequence length
        hidden_emb: same as the hidden dimension of the lstm network

    Returns:
        attn_lstm_emd: same dim as input lstm dim
    """
    def __init__(self, hidden_emb, seq_len=200):
        super(Attention, self).__init__()

        self.seq_len = seq_len
        self.hidden_emb = hidden_emb
        self.mlp1_units = hidden_emb
        self.mlp2_units = hidden_emb

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.hidden_emb, self.mlp1_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.mlp1_units, self.mlp2_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.mlp2_units, self.seq_len),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_emb):

        batch_size = lstm_emb.shape[0]
        lstm_emd = lstm_emb.contiguous()
        lstm_flattened = lstm_emd.view(batch_size, -1)  # (batch, 300 * 2048)

        attn = self.fc(lstm_flattened)                  # (batch, 300) attention over the sequence length
        alpha = self.softmax(attn)

        alpha = torch.stack([alpha] * self.hidden_emb, dim=2)  # (batch, 300, 2048) stack across lstm embedding dim

        attn_lstm_emd = lstm_emd * alpha  # (batch, 300, 2048) give attention weighted lstm embedding
        return attn_lstm_emd


# testing code
if __name__ == '__main__':
    net = Attention(hidden_emb=2048, seq_len=200)
    lstm_emd = torch.Tensor(2, 200, 2048)  
    out = net.forward(lstm_emd)

