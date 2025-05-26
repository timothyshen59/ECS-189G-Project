'''
Concrete IO class for a specific dataset
'''
from sqlite3 import DataError

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import torch
from torchtext.vocab import build_vocab_from_iterator

class Dataset_Loader_Gen(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        self.vocab_size = 0

    def save_vocab(self, path):
        word2idx = dict(self.vocab.get_stoi())
        idx2word = {idx: word for word, idx in word2idx.items()}
        torch.save({
            'word2idx': word2idx,
            'idx2word': idx2word,
            'vocab_size': self.vocab_size
        }, path)

    def load(self):
        print('loading data...')
        jokes = self.load_file()
        jokes = [self.tokenizer(joke) for joke in jokes]
        self.generate_vocab(jokes)
        jokes_indices = self.generate_indices(jokes)
        joke_windows = self.create_window(jokes_indices)
        input_window = []
        target_window = []
        for window in joke_windows:
            input_window.append(window[0])
            target_window.append(window[1])
        padded_input, padded_target = self.pad_windows(input_window, target_window)
        return padded_input, padded_target

    def load_file(self):
        jokes_file_path = '../../data/stage_4_data/text_generation/data'
        jokes = []
        with open(jokes_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = line.split(',', 1) #only splits into max 2 parts (if there are commas in joke itself)
                line = line[1]
                line = line.strip('"')
                jokes.append(line)
        return jokes

    def generate_vocab(self, jokes):
        self.vocab = build_vocab_from_iterator(jokes, specials=['<unk>', '<bos>', '<eos>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.vocab_size = len(self.vocab)

    def generate_indices(self, jokes):
        numerical_jokes = []
        for joke in jokes:
            numerical_tokens = [self.vocab[token] for token in joke]
            numerical_joke = [self.vocab['<bos>']] + numerical_tokens +[self.vocab['<eos>']]
            numerical_jokes.append(numerical_joke)
        return numerical_jokes

    def create_window(self, jokes):
        windows = []
        for joke in jokes:
            for i in range (len(joke) - 5):
                input = joke[i:i+5]
                target = joke[i+1:i+5]
                windows.append((torch.tensor(input), torch.tensor(target)))
        return windows

    def pad_windows(self, input_window, target_window):
        padded_input = pad_sequence(sequences=input_window, batch_first=True, padding_value=self.vocab['<pad>'])
        padded_target = pad_sequence(sequences=target_window, batch_first=True, padding_value=self.vocab['<pad>'])
        return padded_input, padded_target


