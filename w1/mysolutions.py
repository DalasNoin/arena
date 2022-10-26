import torch
from torch.nn.functional import softmax

Q = torch.ones((2,100,64))
K = torch.ones((2,90,64))
V = torch.ones((2,90,64))


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (batch, target sequence length, embedding dim)
    K: shape (batch, source sequence length, embedding dim)
    V: shape (batch, source sequence length, embedding dim)
    softmax(Q KT/sqrt(d_k))V

    Return: shape (same as Q if embedding dim same. batch, target sequence length, output embedding dim)
    '''
    sqrt_d_k = torch.sqrt(torch.tensor(K.shape[-1]))
    query_key = torch.bmm(Q,torch.transpose(K,1,2))
    # print(f"{query_key.shape=} {sqrt_d_k=}")
    result =torch.bmm(softmax(query_key/sqrt_d_k,dim=2), V)
    return result

attention(Q, K, V).shape





Q = torch.ones((2,20,64))
K = torch.ones((2,10,64))
V = torch.ones((2,10,64))

def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    '''
    Should return the results of self-attention.

    You should implement masking for this function. See "The Decoder Side" for an explanation of masking.

    Q: shape (batch, target sequence length, embedding dim)
    K: shape (batch, source sequence length, embedding dim)
    V: shape (batch, source sequence length, embedding dim)
    I = Q K.T
    I.shape = target_len x source_len
    softmax((I+mask)/sqrt(d_k))V

    Return: shape (same as Q if embedding dim same. batch, target sequence length, output embedding dim)
    '''
    sqrt_d_k = torch.sqrt(torch.tensor(K.shape[-1]))
    target_seq_len = torch.tensor(Q.shape[1])
    source_seq_len = torch.tensor(K.shape[1])
    triangular = torch.triu(torch.ones((target_seq_len, source_seq_len), dtype=torch.bool), diagonal=1)
    # print(triangular)

    query_key = torch.bmm(Q, torch.transpose(K,1,2))
    masked_query_key = torch.where(triangular, -torch.inf, query_key)
    # print(masked_query_key)
    result =torch.bmm(softmax((masked_query_key)/sqrt_d_k,dim=2), V)
    return result



result = masked_attention(Q, K, V)

from matplotlib import pyplot as plt

plt.imshow(result.detach().numpy())