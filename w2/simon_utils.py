from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int = 12
    # head_size is not in this config, because in our implementation we're assuming num_heads * head_size = hidden_size
    num_heads: int = 12
    vocab_size: int = 50_257
    # hidden_size is also referred to as embedding_dim, or d_\text{model}d model in some material you might have read.
    hidden_size: int = 768
    # max_seq_len is used just to determine the size of the positional encoding matrix.
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05
    device: str = "cpu"