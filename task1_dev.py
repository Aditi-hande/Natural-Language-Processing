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


# Create vocabulary of words and labels
word_vocab = {}
word_vocab['<PAD>']=0
word_vocab['<UNK>']=1
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
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,collate_fn=collate_fn)


# In[3]:


class BLSTM(nn.Module):
    def __init__(self, embedding_dim, num_lstm_layers, lstm_hidden_dim, lstm_dropout, linear_output_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(lstm_hidden_dim*2, linear_output_dim)
        self.dropout =nn.Dropout(p=lstm_dropout)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_classes)

    def forward(self,data,lengths):
        # Sort the data by length in descending order
        #lengths, sort_indices = torch.sort(lengths, descending=True)
        #data = data[sort_indices]
        # Convert data to embeddings
        embeddings = self.embedding(data)
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
        linear_out = self.activation(self.linear(lstm_out))
        # Pass the output of the linear layer through the classifier
        logits = self.classifier(linear_out)
        return logits
    


# In[4]:


learning_rate = 1
num_epochs = 30
num_classes = len(label_vocab)
num_embeddings = len(word_vocab)
# print(label_vocab)
# print(num_embeddings)



# In[5]:


# Initialize the model
model = BLSTM(embedding_dim=100, num_lstm_layers=1, lstm_hidden_dim=256, lstm_dropout=0.33, linear_output_dim=128, num_classes=num_classes)

#model = torch.load("/content/drive/MyDrive/NLP/data/hw4model.pt")


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight= torch.tensor([1,0.5,1,1,1,1,1,1,1]), ignore_index=-1).cuda()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
lrscheduler =optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,verbose=True)


# In[6]:


#train the model
for epoch in range(num_epochs):
    model.cuda()
    model.train()
    curr_loss=0.0
    for inputs,seqlen,labels in train_dataloader:
        
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs,seqlen)
        #chk shape use .view
        outputs = outputs.view(-1, 9)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_loss+= loss.item()
        
    model.cuda()
    model.eval()
    curr_dev_loss = 0
    with torch.no_grad():
      for val_inputs, val_seqlengths, val_labels in dev_dataloader:
        # val_inputs, val_labels = val_inputs.to(div), val_labels.to(div)
        val_inputs = val_inputs.cuda()
        val_labels = val_labels.cuda()
        
        val_output = model(val_inputs, val_seqlengths)
        val_output = val_output.view(-1, 9)
        val_labels = val_labels.view(-1)
        
        loss = criterion(val_output, val_labels)
        curr_dev_loss += loss    
    # f1 = f1_score(val_labels, val_output, average='micro')
    # print("f1 score is",f1)

    val_loss=curr_dev_loss / len(dev_dataloader)
    lrscheduler.step(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Dev Loss: {val_loss:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {curr_loss/len(train_dataloader):.4f}\n')


# In[9]:


print(outputs.shape)
model_file = 'blstm1.pt'

# save the model
torch.save(model.state_dict(), model_file)

