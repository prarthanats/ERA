import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import re

from collections import Counter
from os.path import exists

from BERT_Model import Transformer
from BERT_Dataset import SentencesDataset
from BERT_Config import get_config

def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

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

def train_model(model, data_loader, optimizer, loss_model, num_iterations, print_each):
    model.train()
    batch_iter = iter(data_loader)
    
    for it in range(num_iterations):
        # Get batch
        batch, batch_iter = get_batch(data_loader, batch_iter)
        masked_input = batch['input'].cuda(non_blocking=True)
        masked_target = batch['target'].cuda(non_blocking=True)
        
        # Forward pass
        output = model(masked_input)
        
        # Compute the cross-entropy loss
        output_v = output.view(-1, output.shape[-1])
        target_v = masked_target.view(-1, 1).squeeze()
        loss = loss_model(output_v, target_v)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print step
        if it % print_each == 0:
            print('it:', it, ' | loss', np.round(loss.item(), 2), ' | Δw:', round(model.embeddings.weight.grad.abs().sum().item(), 3))

def save_embeddings(model, dataset, num_embeddings, values_path, names_path):
    N = num_embeddings
    np.savetxt(values_path, np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    s = [dataset.rvocab[i] for i in range(N)]
    with open(names_path, 'w+') as file:
        file.write('\n'.join(s))

def main():
    config = get_config()
    
    print('loading text...')
    sentences = load_sentences('training.txt')
    
    print('tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = tokenize_sentences(sentences, special_chars)
    
    print('creating/loading vocab...')
    vocab = create_or_load_vocab(sentences, 'vocab.txt', config['n_vocab'])
    
    print('creating dataset...')
    dataset = SentencesDataset(sentences, vocab, config['seq_len'])
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, drop_last=True, pin_memory=True, batch_size=config['batch_size'])
    
    print('initializing model...')
    inner_ff_size = config['embed_size'] * 4
    model = Transformer(config['n_code'], config['n_heads'], config['embed_size'], inner_ff_size, len(dataset.vocab), config['seq_len'], config['dropout']).cuda()
    
    print('initializing optimizer and loss...')
    optim_kwargs = config["optim_kwargs"]
    optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)
    
    print('training...')
    print_each = 10
    num_iterations = 10000
    train_model(model, data_loader, optimizer, loss_model, num_iterations, print_each)
    
    print('saving embeddings...')
    save_embeddings(model, dataset, 3000, 'values.tsv', 'names.tsv')
    
    print('end')

if __name__ == "__main__":
    main()