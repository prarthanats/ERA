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



#### Implementation and Inference Details  


#### Training Loss Plot


#### Training Logs
   
