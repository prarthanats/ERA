# English-to-French Sentence Translation using Transformers

The goal of this assignment was to train an English to French translation model using the OPUS book translation dataset using the PyTorch Lightening framework.

## Requirements
1. Speeding up Transformers
2. Achieve a final loss of less than 1.8 during training

## Introduction to Data - [Opus Books](https://huggingface.co/datasets/opus_books)
   
The opus_books dataset provided by Hugging Face is a part of the OPUS (Open Parallel Corpus) project, which aims to collect and provide high-quality parallel corpora. This dataset specifically focuses on parallel texts from books, making it a valuable resource for training and evaluating machine translation models, among other natural language processing tasks.

### Dataset Description:

Source: The dataset consists of parallel texts from various books, making it a diverse and extensive collection of text data.
Languages: It contains parallel texts in multiple languages, allowing you to work on a wide range of language pairs.
Content: The dataset is primarily focused on books, which can include various genres and topics, making it suitable for a variety of natural language understanding and generation tasks.

### Usage:

- Machine Translation: You can use this dataset to train and evaluate machine translation models. Given its diversity of languages and topics, it can be valuable for building translation systems that work across different domains.
- Language Understanding: Beyond translation, you can leverage this dataset for various natural language understanding tasks, such as text classification, sentiment analysis, and more.
- Research: The dataset can be utilized for research purposes, including multilingual and cross-lingual studies, as well as for developing and testing novel NLP algorithms.

### Sample Data Sample for English to French

![image](https://github.com/prarthanats/ERA/assets/32382676/d8d9560d-3223-4f62-a158-3bff5a4c0f31)

### Description of the Code 

[Assignment 15 ](https://github.com/prarthanats/ERA/blob/main/S15_Assignment/README.md)

### Changes for speeding up the Code from Assignment 15

#### Data Source 

Dataset Used is [English-French](https://huggingface.co/datasets/opus_books/viewer/en-fr/train)

![image](https://github.com/prarthanats/ERA/assets/32382676/bdf9e70b-de48-4436-bbf8-f18b23e4585d)

#### Data Preprocessing 

1. Removed English sentences that exceeded 150 characters
2. Removed French sentences whose length was greater than English sentence length plus 10 characters
3. Removed all sentences whose length is less than 2 characters
   
![image](https://github.com/prarthanats/ERA/assets/32382676/5d772323-1b6b-485b-a44e-f89499b924b8)

4. Implemented dynamic padding to handle variable-length sequences for each batch efficiently.

This function takes a batch of data samples, calculate the lengths of the encoder inputs and decoder inputs for each sample in the batch and store them in encode_batch_length and decode_batch_length lists, respectively.Then finds the maximum sequence length among all encoder and decoder inputs in the batch (max_seq_len), and add 2 to it. This is done to ensure that there is room for additional tokens if needed. Iterate over each item (data sample) in the batch (bat):

   a. Pad the encoder_input, decoder_input, and label tensors with the pad_token to match the maximum sequence length (max_seq_len). This is done using torch.cat to concatenate the original tensor with a tensor filled with pad_token values for the required number of elements.
   
   b. Append the source and target texts of the data sample to the src_text and tgt_text lists, respectively.

~~~
encode_batch_length = list(map(lambda x : x['encoder_input'].size(0),bat))
decode_batch_length = list(map(lambda x : x['decoder_input'].size(0),bat))
max_seq_len = max( encode_batch_length +  decode_batch_length)
max_seq_len = max_seq_len + 2

for item in bat:
    item['encoder_input'] = torch.cat([item['encoder_input'],
                                       torch.tensor([pad_token] * (max_seq_len-item['encoder_input'].size(0)), dtype = torch.int64),],dim=0)
    item['decoder_input'] = torch.cat([item['decoder_input'],
                                    torch.tensor([pad_token] * (max_seq_len-item['decoder_input'].size(0)), dtype = torch.int64),],dim=0)
    
    item['label'] = torch.cat([item['label'],
                                    torch.tensor([pad_token] * (max_seq_len-item['label'].size(0)), dtype = torch.int64),],dim=0)

    src_text.append(item['src_text'] )
    tgt_text.append(item['tgt_text'] )
~~~

#### Model Architecture

1. Encoder-Decoder: This model utilizes the encoder-decoder architecture for sequence-to-sequence translation tasks.
2. Parameter Sharing: Implemented parameter sharing between encoder and decoder block to reduce the number of parameters
~~~
Sharing Pattern :
[e1, e2, e3, e1, e2, e3] - for encoder
[d1, d2, d3, d1, d2, d3] - for decoder
~~~
3. Dense feedforward layer size (dff) reduced to 128

#### Setup and Usage

The model can be run using Lightning framework as shown below:

Clone the GitHub Repo
~~~
https://github.com/prarthanats/ERA/tree/af9cd46c1fc13d453a4875ce87ecd7f2148ffac2/S16_Assignment
~~~

Download and Run the 

~~~
https://github.com/prarthanats/ERA/blob/main/S16_Assignment/Assignment_16_Final_30%20Epochs.ipynb
~~~

#### Implementation and Inference Details  

~~~
Epochs - 30
Batch Size - 128
Number of parameters: 56.3 M  
Total estimated model params size - 225.350MB
Final training loss - 1.61
~~~

#### Training Logs

~~~
Training: 0it [00:00, ?it/s]
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  0 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- C ' est la porte de la porte , dit - t - t - il , et le de la tête de la tête .
TRAINING LOSS :  6.34243
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  1 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains de la main , vous savez , s ' écria Buckingham , en riant , en riant , sans doute , sans doute , sans doute , sans doute , sans doute .
TRAINING LOSS :  4.79916
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  2 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Au fond de la main , vous voulez dire à Buckingham , en faisant un coup de l ' on se , sans cesse , sans cesse .
TRAINING LOSS :  4.06622
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  3 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire Buckingham , en levant sa voix , en s ' efforçant de s ' occuper de son peuple , sans tout cela se plaît .
TRAINING LOSS :  3.60775
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  4 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire , s ' écria Buckingham en levant la voix afin de faire la attention , sans qu ' il y a absolument .
TRAINING LOSS :  3.31004
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  5 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous entendez ! cria Buckingham en levant la voix pour attirer la voix de ses gens sans s ' en criant .
TRAINING LOSS :  3.09803
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  6 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous entendez ! s ' écria Buckingham en levant la voix , en levant la voix de son peuple , sans rien jeter .
TRAINING LOSS :  2.92758
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  7 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire ! s ' écria Buckingham en élevant la voix pour attirer les attention de son peuple , sans criant qu ' il arrivât tout prix .
TRAINING LOSS :  2.78572
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  8 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire ! s ' écria Buckingham en haussant la voix pour attirer l ' attention de son peuple , sans crier .
TRAINING LOSS :  2.66375
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  9 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire , dit Buckingham en haussant la voix pour plaire à son peuple , sans crier .
TRAINING LOSS :  2.52762
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  10 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire ! s ' écria Buckingham en relevant sa voix pour attirer la garde de ses gens , sans se jeter comme à lui .
TRAINING LOSS :  2.35945
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  11 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire , dit Buckingham en relevant la voix de ses gens , sans s ' arreter .
TRAINING LOSS :  2.22039
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  12 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Au diable , vous voulez dire , s ' écria Buckingham en levant sa voix pour attirer l ' attention de ses gens sans qu ' il .
TRAINING LOSS :  2.10459
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  13 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous entendez !» s ' écria Buckingham en relevant la voix de son peuple , sans jeter des cris comme pour s ' en .
TRAINING LOSS :  2.00441
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  14 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire , dit Buckingham en levant la voix pour attirer la moindre de ses gens sans s ' arreter .
TRAINING LOSS :  1.91825
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  15 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Au diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre attention de ses gens , sans en criant .
TRAINING LOSS :  1.84494
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  16 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Vous voulez dire du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer l ' attention de ses gens sans en criant .
TRAINING LOSS :  1.78321
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  17 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la attention de ses gens , sans en criant .
TRAINING LOSS :  1.73349
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  18 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la attention de ses gens sans crier .
TRAINING LOSS :  1.70175
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  19 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant la voix pour attirer la attention de ses gens sans en criant .
TRAINING LOSS :  1.68746
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  20 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre attention de ses gens sans crier .
TRAINING LOSS :  1.67653
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  21 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre attention de ses gens sans crier .
TRAINING LOSS :  1.66613
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  22 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant la voix pour attirer l ' attention de ses gens sans s ' en criant .
TRAINING LOSS :  1.65748
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  23 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre attention de ses gens sans crier .
TRAINING LOSS :  1.64995
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  24 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant la voix pour attirer la moindre de ses gens , sans en criant .
TRAINING LOSS :  1.6429
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  25 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre de ses gens , sans en criant .
TRAINING LOSS :  1.63735
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  26 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre de ses gens , sans en criant .
TRAINING LOSS :  1.63158
TIME TAKEN (minutes): 0.18
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  27 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre attention de ses gens sans crier .
TRAINING LOSS :  1.62656
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  28 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant la voix pour attirer la moindre de ses gens , sans en criant .
TRAINING LOSS :  1.62257
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
Validation: 0it [00:00, ?it/s]
----------------------------------------------------------------------
EPOCH :  29 :
SOURCE : "In the hands of the devil, you mean!" cried Buckingham, raising his voice so as to attract the notice of his people, without absolutely shouting.
EXPECTED  : -- Dans les mains du diable, vous voulez dire, s'écria Buckingham en élevant la voix pour attirer du monde, sans cependant appeler directement.
PREDICTED : -- Dans les mains du diable , vous voulez dire !» s ' écria Buckingham en levant sa voix pour attirer la moindre attention de ses gens sans crier .
TRAINING LOSS :  1.61962
TIME TAKEN (minutes): 0.19
----------------------------------------------------------------------
INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=30` reached.
~~~
