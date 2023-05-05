import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from transformers import AlbertTokenizer, AlbertModel
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from typing import Tuple
import json
import argparse
from datasets import Dataset, ClassLabel, Features, Value


def load_dataset_gossip(datapath='./fakenewsnet_dataset'):
    data = []
    labels = []
    real_paths = os.listdir(os.path.join(datapath, 'gossipcop', 'real'))
    fake_paths = os.listdir(os.path.join(datapath, 'gossipcop', 'fake'))

    for path in real_paths:
        cur_path = os.path.join(datapath, 'gossipcop', 'real', path, r'news content.json')
        try:
            text = json.loads(open(cur_path).read())['text']
        except:
            continue
        data.append(text)
        labels.append(1)

    for path in fake_paths:
        cur_path = os.path.join(datapath, 'gossipcop', 'fake', path, r'news content.json')
        try:
            text = json.loads(open(cur_path).read())['text']
        except:
            continue
        data.append(text)
        labels.append(0)

    return np.array(data), np.array(labels)


def load_dataset_political(datapath='./fakenewsnet_dataset'):
    data = []
    labels = []
    real_paths = os.listdir(os.path.join(datapath, 'politifact', 'real'))
    fake_paths = os.listdir(os.path.join(datapath, 'politifact', 'fake'))

    for path in real_paths:
        # print(os.path.exists(os.path.join(datapath, 'politifact', 'real', path, r'news content.json')))
        cur_path = os.path.join(datapath, 'politifact', 'real', path, r'news content.json')
        # print(json.loads(open(cur_path).read())['text'])
        try:
            text = json.loads(open(cur_path).read())['text']
        except:
            continue
        data.append(text)
        labels.append(1)

    for path in fake_paths:
        cur_path = os.path.join(datapath, 'politifact', 'fake', path, r'news content.json')
        try:
            text = json.loads(open(cur_path).read())['text']
        except:
            continue
        data.append(text)
        labels.append(0)


    return np.array(data), np.array(labels)


def load_data(dataname='fake'):
    if dataname=='imdb':
        dataset = load_dataset('imdb')
        train_dataset = dataset['train'].select(range(5000))
        # val_dataset = dataset['test'].select(range(500))
        yelp_dataset = load_dataset('yelp_polarity')
        val_dataset = yelp_dataset['test']
    elif dataname=='fake':
        text_train, train_label = load_dataset_gossip()
        text_test, test_label = load_dataset_political()
        # print(text_test)
        features = Features({'text': Value('string'), 'label': ClassLabel(num_classes=2, names=[1, 0])})
        train_dataset = Dataset.from_dict({'text': text_train, 'label': train_label}, features=features)
        val_dataset = Dataset.from_dict({'text': text_test, 'label': test_label}, features=features)

    return train_dataset, val_dataset


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

train_dataset, val_dataset = load_data()

# set hyperparameters
vocab_size = 50000
embedding_dim = 100
n_filters = 100
filter_sizes = [3, 4, 5]
output_dim = 2
dropout = 0.5
lr = 1e-3
batch_size = 64
epochs = 500

# load data
train_data = train_dataset['text']
train_labels = train_dataset['label']
val_data = val_dataset['text']
val_labels = val_dataset['label']

# preprocess data
tokenizer = lambda x: x.split()
train_data = [tokenizer(sentence) for sentence in train_data]
val_data = [tokenizer(sentence) for sentence in val_data]

# build vocabulary
from collections import Counter
word_count = Counter()
for sentence in train_data:
    word_count.update(sentence)
vocab = [word for word, count in word_count.most_common(vocab_size)]

# create word to index mapping
word_to_idx = {word: i+2 for i, word in enumerate(vocab)}
word_to_idx['<pad>'] = 0
word_to_idx['<unk>'] = 1

# convert words to indices
train_data = [[word_to_idx.get(word, 1) for word in sentence] for sentence in train_data]
val_data = [[word_to_idx.get(word, 1) for word in sentence] for sentence in val_data]

# pad sequences
max_len = max(len(sentence) for sentence in train_data)
max_len = 10
train_data = [sentence[:max_len] + [0] * (max_len - len(sentence)) for sentence in train_data]
val_data = [sentence[:max_len] + [0] * (max_len - len(sentence)) for sentence in val_data]

# convert to tensors
train_data = torch.LongTensor(train_data)
train_labels = torch.LongTensor(train_labels)
val_data = torch.LongTensor(val_data)
val_labels = torch.LongTensor(val_labels)


train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

for epoch in range(epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.shape)
        loss = criterion(output, target)
        # print(loss)
        loss.backward()
        optimizer.step()
        print(loss)
        train_loss += loss.item() * data.size(0)
        # print(loss)
        _, preds = torch.max(output, 1)
        train_acc += torch.sum(preds == target)
    train_loss /= len(train_loader.dataset)
    train_acc = train_acc.float() / len(train_loader.dataset)
    print(train_acc)
val_loss = 0
val_acc = 0
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item() * data.size(0)
        _, preds = torch.max(output, 1)
        val_acc += torch.sum(preds == target)
val_loss /= len(val_loader.dataset)
val_acc = val_acc.float() / len(val_loader.dataset)

print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
