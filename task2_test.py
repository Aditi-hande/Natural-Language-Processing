#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np



# In[2]:


class NERDataset(Dataset):
    def __init__(self, filename, word_vocab, label_vocab):
        self.data = []
        with open(filename, 'r') as f:
            sentence = []
            labels = []
            for line in f:
                line = line.strip()
                if not line: #blank line
                    self.data.append((sentence, labels))
                    sentence = []
                    labels = []
                else:
                    fields = line.split()
                    word = fields[1]
                    label = fields[-1]
                    if(word in word_vocab):
                      sentence.append(word_vocab[word])
                      labels.append(label_vocab[label])
                    else:
                      sentence.append(1)
                      labels.append(label_vocab[label])
        self.data.append((sentence, labels))

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence, labels = self.data[index]
        return torch.tensor(sentence), torch.tensor(labels)

div = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create vocabulary of words and labels
word_vocab = {}
word_vocab['<PAD>']=0
word_vocab['unk']=1
label_vocab = {}

with open('data/train', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            fields = line.split()
            word = fields[1]
            label = fields[-1]
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)

inverted_label_vocab = {v: k for k, v in label_vocab.items()}
word_list = list(word_vocab.keys())


# Load train, dev and test datasets
train_dataset = NERDataset('data/train', word_vocab, label_vocab)
dev_dataset = NERDataset('data/dev', word_vocab, label_vocab)

def collate_fn(batch):
    inputs,target = zip(*batch)
    seq_length = torch.Tensor([len(seq) for seq in inputs])
    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first = True, padding_value = 0)
    padded_targets = nn.utils.rnn.pad_sequence(target, batch_first = True, padding_value = -1)
    return padded_inputs,seq_length,padded_targets
    

# Create dataloaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,collate_fn=collate_fn)


# In[3]:


glove_embedding = {}

with open('glove.6B.100d.txt',encoding='utf-8', mode='r') as f:
  for line in f:
    fields=line.split()
    word = fields[0]
    embedding = np.asarray(fields[1:], dtype='float32')
    glove_embedding[word]=embedding


# In[4]:


word_embedding =[]

for word in word_list:
  lower_word = word.lower()
  if word[0].isupper(): #capital append with 1
    if lower_word in glove_embedding:
      temp_embedding = np.concatenate([glove_embedding[lower_word],np.asarray([1])],axis=0)
    else:
      temp_embedding = np.concatenate([glove_embedding['unk'],np.asarray([1])],axis=0)
#       temp_embedding = np.concatenate((np.random.rand(100), np.asarray([1])), axis=0)


  else: # non capital append with 0
    if lower_word in glove_embedding:
      temp_embedding = np.concatenate([glove_embedding[lower_word],np.asarray([0])],axis=0)
    else:
      temp_embedding = np.concatenate([glove_embedding['unk'], np.asarray([0])], axis=0)
       #temp_embedding = np.concatenate((np.random.rand(100), np.asarray([0])), axis=0)



  word_embedding.append(temp_embedding)


# In[5]:


class BLSTM(nn.Module):
    def __init__(self, embedding_dim, num_lstm_layers, lstm_hidden_dim, lstm_dropout, linear_output_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), padding_idx = 0, freeze = False)
        #self.embedding = nn.Embedding(num_embeddings, embedding_dim,padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_dim*2, linear_output_dim)
        self.dropout =nn.Dropout(p=lstm_dropout)
        self.dropout1 =nn.Dropout(p=0.2)
        self.dropout2 =nn.Dropout(p=0.2)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_classes)

    def forward(self,data,lengths):
        # Sort the data by length in descending order
        #lengths, sort_indices = torch.sort(lengths, descending=True)
        #data = data[sort_indices]
        # Convert data to embeddings
        embeddings = self.embedding(data)
        embeddings = self.dropout2(embeddings)
        # Pack the padded sequence
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        # Pass the packed sequence through the LSTM
        lstm_out, _ = self.lstm(packed_embeddings)
        # Unpack the padded sequence
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # Re-sort the data back to the original order
        # _, unsort_indices = torch.sort(sort_indices)
        #lstm_out = lstm_out[unsort_indices]
        # Pass the output of the LSTM through the linear layer and activation function
        #dropout
        lstm_out = self.dropout(lstm_out)
        linear_out = self.activation(self.dropout1(self.linear(lstm_out)))
        # Pass the output of the linear layer through the classifier
        logits = self.classifier(linear_out)
        return logits
    


# In[6]:


learning_rate = 1
num_epochs = 50
num_classes = len(label_vocab)
num_embeddings = len(word_vocab)
# print(label_vocab)
# print(num_embeddings)



# In[11]:


model_loaded = BLSTM(embedding_dim=101, num_lstm_layers=1, lstm_hidden_dim=256, lstm_dropout=0.33, linear_output_dim=128, num_classes=num_classes)

# Load the state dictionary from the saved model
state_dict = torch.load("blstm2.pt", map_location=torch.device('cpu'))

# Load the state dictionary into the model
model_loaded.load_state_dict(state_dict)
model_loaded.eval()

sentence_list = []
label_list =[]
f = open('data/dev', 'r')
sentence = []
labels = []
for line in f:
  line = line.strip()
  if not line: #blank line
      sentence_list.append(sentence)
      label_list.append(labels)
      sentence = []
      labels = []
  else:
      fields = line.split()
      word = fields[1]
      label = fields[-1]
      sentence.append(word)
      labels.append(label)
        
sentence_list.append(sentence)
label_list.append(labels)

f.close


f = open('dev2.out','w')


for i, s in enumerate(sentence_list):
  gold_tag = label_list[i]
  idx_word = torch.tensor([[word_vocab[word] if word in word_vocab else word_vocab['unk'] for word in s]])
  lengths =[len(s)]


  with torch.no_grad():
    output = model_loaded(idx_word,lengths)
    output = output.view(-1, 9)
  _, predicted = torch.max(output, 1)
  predicted_labels = [inverted_label_vocab[i.item()] for i in predicted]

  idx = 0

  for fileword, filelabel, filegold in zip(s, predicted_labels,gold_tag):
    idx +=1
    filestring = str(idx) + " " + str(fileword) + " "  + str(filelabel) + "\n"
    f.write(filestring)

  f.write("\n")

f.close


# In[10]:


sentence_list_test = []

f = open('data/test', 'r')
sentence_test = []
for line in f:
  line = line.strip()
  if not line: #blank line
      sentence_list_test.append(sentence_test)
      sentence_test = []
  else:
      fields = line.split()
      word = fields[1]
      sentence_test.append(word)
        
sentence_list_test.append(sentence_test)
f.close


f = open('test2.out','w')


for i, s in enumerate(sentence_list_test):
  
  idx_word_test = torch.tensor([[word_vocab[word] if word in word_vocab else word_vocab['unk'] for word in s]])
  lengths_test =[len(s)]


  with torch.no_grad():
    output_test = model_loaded(idx_word_test,lengths_test)
    output_test = output_test.view(-1, 9)
  _, predicted = torch.max(output_test, 1)
  predicted_labels = [inverted_label_vocab[i.item()] for i in predicted]

  idx = 0

  for fileword, filelabel in zip(s, predicted_labels):
    idx +=1
    filestring = str(idx) + " " + str(fileword) +  " " + str(filelabel) + "\n"
    f.write(filestring)

  f.write("\n")

f.close


# In[ ]:




