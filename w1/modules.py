import torch
from torch import nn, Tensor


# more efficient, buffer for pe version
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device:str="cpu"):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        L = self.max_len
        partial_term = torch.outer(torch.arange(L),1/10_000**(torch.arange(torch.ceil(torch.tensor(self.d_model/2)))*2/self.d_model))
        positional_encoding = torch.zeros((L, self.d_model)).to(device)
        positional_encoding[:,::2] = torch.sin(partial_term.to(device))
        positional_encoding[:,1::2] = torch.cos(partial_term.to(device))
        self.register_buffer("positional_encoding", positional_encoding)


    def forward(self, x: Tensor) -> Tensor:
        '''
        x: Tensor, shape [batch, seq_len, embedding_dim]
        '''
        L = x.shape[1]

        return self.dropout(x + self.positional_encoding[:L,:])