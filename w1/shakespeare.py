
vocab_size = 27687 # tokenizer.vocab_size
from torch.nn.functional import one_hot
import torch
from torch.utils.data import Dataset
import re
from nltk.tokenize import word_tokenize


def get_shakespeare() -> list:
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
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

def get_shakespeare_word_tokenizer():
    """
    returns list of list of indices, vocabulary
    """
    path = "100-0.txt"
    tokens = []
    with open(path,"r", encoding="utf8") as f:
        while True:
        
            # Get next line from file
            line = f.readline().lower()
            line = re.sub("\d+", "", line)
            line = line.encode("latin", "ignore") 
            line = line.decode("utf-8", "ignore") 
            delete_these = ["'",'"','-', '_', '@', '.',
            ',','\\','/','/','!','#','%','&','(',')',';',':',"?",
            '$']
            for char in delete_these:
                line = line.replace(char, " ")


        
            # if line is empty
            # end of file is reached
            if not line:
                break
            tokenized = word_tokenize(line)
            # tokenized = tokenizer(line,padding=True,
            #                              max_length=128,
            #                              # padding='max_length',
            #                              truncation=True)["input_ids"]
            if len(tokenized) > 2:
                tokens.append(tokenized)
    all_unique_tokens = []
    for token_sentence in tokens:
        for token in token_sentence:
            if token in all_unique_tokens:
                continue
            all_unique_tokens.append(token)
    
    token_indices = []
    for token_sentence in tokens:
        token_sentence_indices = []
        for token in token_sentence:
            token_sentence_indices.append(all_unique_tokens.index(token))
        token_indices.append(token_sentence_indices)
    return token_indices, all_unique_tokens

class ShakespeareDataset(Dataset):
    # def __init__(self, text, labels):
    #     self.labels = labels
    #     self.text = text
    def __init__(self, config, use_word_tokenizer:bool=True):
        self.config = config
        self.vocab_size = vocab_size
        if use_word_tokenizer:
            self.tokens, self.vocabulary = get_shakespeare_word_tokenizer()
            self.vocab_size = len(self.vocabulary)
            # assert vocab_size==self.vocab_size, "please change vocab size manually"
        else:
            self.tokens = get_shakespeare()
        self.device = config.device
        self.total_size = len(self.tokens)
        # self.text = torch.ones(self.total_size,
        #                         self.seq_len,
        #                         config.hidden_size)
    
    def decode(self, indices: list) -> list:
        output = ' '.join([self.vocabulary[index] for index in indices])
        return output

    def __len__(self):
            return self.total_size

    def __getitem__(self, idx):
            tokens = self.tokens[idx]
            # one_hot_matrix = one_hot(torch.Tensor(tokens,device=self.device).to(torch.int64), num_classes=self.vocab_size)
            tokens = torch.tensor(tokens).to(torch.int64).to(self.device)
            
            text = tokens[:-1]
            label = tokens[1:]
            # sample = {"text": text, "label": label}
            return text, label

class DummyTokenizer():
    """
    I know this code is very messy and repeated just above, but the smapler expected a tokenizer class. so i just copypasted this together.
    """
    
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def encode(self, text):
        line = text.lower()
        line = re.sub("\d+", "", line)
        line = line.encode("latin", "ignore") 
        line = line.decode("utf-8", "ignore") 
        delete_these = ["'",'"','-', '_', '@', '.',
        ',','\\','/','/','!','#','%','&','(',')',';',':',"?",
        '$']
        for char in delete_these:
            line = line.replace(char, " ")
        
        tokenized = word_tokenize(line)
        token_sentence_indices = []
        for token in tokenized:
            token_sentence_indices.append(self.vocabulary.index(token))
        return token_sentence_indices

    def decode(self, indices):
        return ' '.join([self.vocabulary[index] for index in indices])


if __name__=="__main__":
    result, vocab = get_shakespeare_word_tokenizer()
    print(result[0], vocab[-200:], len(vocab))