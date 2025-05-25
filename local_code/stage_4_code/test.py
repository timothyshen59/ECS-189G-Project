from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
import torch

with open('../../data/stage_4_data/text_classification/train/neg/1_1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokenizer = get_tokenizer('basic_english')
tokens = tokenizer(text)
tokens = [word for word in tokens if word.isalpha()]
print(tokens)

glove = GloVe(name='6B', dim=100)
indices = []
for token in tokens:
    if token in glove.stoi:
        indices.append(glove.stoi[token])
    else:
        #indices.append(1)
        print(token)
print(indices)

embeddings = glove.vectors[torch.tensor(indices)]
print(embeddings.shape)