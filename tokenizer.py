import re
import torch
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self, max_len: int = 512):
        self.vocab: Dict[str, int] = {}
        self.max_len = max_len
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1
        }
        self.vocab_size = len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        text = ' '.join(text.split())
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        word_freq = {}
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        return self.vocab_size

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        tokens = tokens[:self.max_len]
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        if len(ids) < self.max_len:
            ids.extend([self.vocab[self.pad_token]] * (self.max_len - len(ids)))
        return ids

    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        encoded = [self.encode(text) for text in texts]
        return torch.tensor(encoded)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_collate_fn(self):
        def collate_fn(batch):
            texts, labels = zip(*batch)
            encoded_texts = self.batch_encode(texts)
            labels = torch.tensor(labels)
            return encoded_texts, labels
        return collate_fn


