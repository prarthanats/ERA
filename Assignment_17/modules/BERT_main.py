import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import re

from collections import Counter
from os.path import exists
from config import get_bert_config

from torch.utils.data import Dataset
import random
import torch


class BERTDataset(Dataset):
    #Init dataset
    def __init__(self, sentences, vocab, seq_len):
        dataset = self
        
        dataset.sentences = sentences
        dataset.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.rvocab = {v:k for k,v in dataset.vocab.items()}
        dataset.seq_len = seq_len
        
        #special tags
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
    
    
    #fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self
        
        #while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask
        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]
        
        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    #return length
    def __len__(self):
        return len(self.sentences)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s] 
        return s

def load_sentences(file_path):
    with open(file_path) as file:
        return file.read().lower().split('\n')

def tokenize_sentences(sentences, special_chars):
    tokenized_sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    return [[w for w in s if len(w)] for s in tokenized_sentences]

def create_or_load_vocab(sentences, vocab_path, max_vocab_size):
    if not exists(vocab_path):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(max_vocab_size)
        vocab = [w[0] for w in vocab]
        with open(vocab_path, 'w+') as file:
            file.write('\n'.join(vocab))
    else:
        with open(vocab_path) as file:
            vocab = file.read().split('\n')
    return vocab

def get_batch_bert(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def train_model_bert(model, data_loader, optimizer, loss_model, num_iterations, print_each):
    print_each = 10
    model.train()
    batch_iter = iter(data_loader)
    n_iteration = 100
    for it in range(n_iteration):

    #get batch
      batch, batch_iter = get_batch_bert(data_loader, batch_iter)

      #infer
      masked_input = batch['input']
      masked_target = batch['target']

      masked_input = masked_input.cuda(non_blocking=True)
      masked_target = masked_target.cuda(non_blocking=True)
      output = model(masked_input)

      #compute the cross entropy loss
      output_v = output.view(-1,output.shape[-1])
      target_v = masked_target.view(-1,1).squeeze()
      loss = loss_model(output_v, target_v)

      #compute gradients
      loss.backward()

      #apply gradients
      optimizer.step()

      #print step

      if it % print_each == 0:
          print('it:', it,
                ' | loss', np.round(loss.item(),2),
                ' | Δw:', round(model.embeddings.weight.grad.abs().sum().item(),3))



def save_embeddings(model, dataset, num_embeddings, values_path, names_path):
    N = num_embeddings
    np.savetxt(values_path, np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    s = [dataset.rvocab[i] for i in range(N)]
    with open(names_path, 'w+') as file:
        file.write('\n'.join(s))