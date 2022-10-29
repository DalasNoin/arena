from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
from torch.nn.functional import one_hot
import torch
from torch.utils.data import Dataset

def get_shakespeare():
    path = "100-0.txt"
    tokens = []
    with open(path,"r", encoding="utf8") as f:
        while True:
        
            # Get next line from file
            line = f.readline()
        
            # if line is empty
            # end of file is reached
            if not line:
                break
            tokenized = tokenizer(line,padding=True,
                                         max_length=128,
                                         # padding='max_length',
                                         truncation=True)["input_ids"]
            if len(tokenized) > 2:
                tokens.append(tokenized)
    return tokens

class ShakespeareDataset(Dataset):
    # def __init__(self, text, labels):
    #     self.labels = labels
    #     self.text = text
    def __init__(self, config):
        self.config = config
        self.vocab_size = vocab_size
        self.tokens = get_shakespeare()
        self.device = config.device
        self.total_size = len(self.tokens)
        # self.text = torch.ones(self.total_size,
        #                         self.seq_len,
        #                         config.hidden_size)

    def __len__(self):
            return self.total_size

    def __getitem__(self, idx):
            tokens = self.tokens[idx]
            # one_hot_matrix = one_hot(torch.Tensor(tokens,device=self.device).to(torch.int64), num_classes=self.vocab_size)
            tokens = torch.Tensor(tokens,device=self.device).to(torch.int64)
            
            text = tokens[:-1]
            label = tokens[1:]
            # sample = {"text": text, "label": label}
            return text, label



if __name__=="__main__":
    result = get_shakespeare()
    print(result[0])