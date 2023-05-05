# Import necessary libraries
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
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fake',
                    choices=['fake','imdb'])
parser.add_argument('--test_time', action='store_true', default=False,
                    help='Test time.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight', type=float, default=0.5,
                    help='SSL weight.')
args = parser.parse_known_args()[0]

# set seeds
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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


def load_data(dataname='imdb'):
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
# Load IMDB dataset

train_dataset, val_dataset = load_data(dataname=args.dataset)

# Select a portion of the train and test datasets

# Load ALBERT tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')

# Set the number of output classes
num_classes = 2

# Freeze all the parameters of the model except for the classifier layer
for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
for param in model.classifier.parameters():
    param.requires_grad = True

# Prepare the dataset for training
train_encodings = tokenizer(train_dataset['text'], truncation=True, padding=True)
train_labels = torch.Tensor(train_dataset['label'])
train_dataset = list(zip(torch.Tensor(train_encodings['input_ids']).to(dtype=torch.int32), torch.Tensor(train_encodings['attention_mask']).to(dtype=torch.int32), train_labels.to(dtype=torch.long)))

val_encodings = tokenizer(val_dataset['text'], truncation=True, padding=True)
val_labels = torch.Tensor(val_dataset['label'])
val_dataset = list(zip(torch.Tensor(val_encodings['input_ids']).to(dtype=torch.int32), torch.Tensor(val_encodings['attention_mask']).to(dtype=torch.int32), val_labels.to(dtype=torch.long)))

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define the training loop
def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for input_ids, attention_mask, labels in tqdm(dataloader, desc = 'Training'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = model.classifier(outputs.last_hidden_state[:, 0, :])
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

# Define the evaluation loop
def eval_loop(dataloader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc = 'Evaling'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = model.classifier(outputs.last_hidden_state[:, 0, :])
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Define the training parameters
batch_size = 32
num_epochs = 3

# Prepare the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loop(train_loader, model, loss_fn, optimizer, device)
    #train_accuracy = eval_loop(train_loader, model, device)
    val_accuracy = eval_loop(val_loader, model, device)
    #print(f"Training loss: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

with open(f"./{args.dataset}_results.txt", "a+") as f:
    f.write(f"This is not ssl. Validation accuracy: {val_accuracy:.4f}, dataset: {args.dataset}, test_time: {args.test_time}, weight: {args.weight}, lr: {args.lr}, weight_decay: {args.weight_decay}, mask_ratio: {args.mask_ratio}\n")
