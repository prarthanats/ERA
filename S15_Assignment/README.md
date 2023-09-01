
# English-to-Italian Sentence Translation using Transformers

This repository contains code and utilities for training and evaluating deep learning models, for sentence translation using the PyTorch Lightening framework.

## Requirements

1. Rewrite the whole code covered in the class in Pytorch-Lightning (code copy will not be provided)
2. Train the model for 10 epochs
3. Achieve a loss of less than 4. Loss should start from 9-10, and reduce to 4, showing that your code is working

## Introduction to Data - [Opus Books]('https://huggingface.co/datasets/opus_books')

The opus_books dataset provided by Hugging Face is a part of the OPUS (Open Parallel Corpus) project, which aims to collect and provide high-quality parallel corpora. This dataset specifically focuses on parallel texts from books, making it a valuable resource for training and evaluating machine translation models, among other natural language processing tasks.

### Dataset Description:

Source: The dataset consists of parallel texts from various books, making it a diverse and extensive collection of text data.
Languages: It contains parallel texts in multiple languages, allowing you to work on a wide range of language pairs.
Content: The dataset is primarily focused on books, which can include various genres and topics, making it suitable for a variety of natural language understanding and generation tasks.

### Usage:

Machine Translation: You can use this dataset to train and evaluate machine translation models. Given its diversity of languages and topics, it can be valuable for building translation systems that work across different domains.
Language Understanding: Beyond translation, you can leverage this dataset for various natural language understanding tasks, such as text classification, sentiment analysis, and more.
Research: The dataset can be utilized for research purposes, including multilingual and cross-lingual studies, as well as for developing and testing novel NLP algorithms.

### Accessing the Dataset:

You can access the opus_books dataset through the Hugging Face Datasets library. It provides easy-to-use APIs for downloading and working with the data in your machine learning projects.
You can explore and download the dataset from the Hugging Face website, where additional information and usage examples may be available.

![image](https://github.com/prarthanats/ERA/assets/32382676/bb74083d-c064-4676-8367-84744075a308)

## Model Architecture
~~~
Start
|
|--- InputEmbeddings
|       |--- nn.Embedding
|--- PositionalEncoding
|       |--- Initialization
|       |--- Positional Encoding Calculation
|--- LayerNormalization
|       |--- Initialization
|       |--- Forward Calculation
|--- MultiHeadAttentionBlock
|       |--- Initialization
|       |--- Attention Calculation
|--- FeedForwardBlock
|       |--- Initialization
|       |--- Forward Calculation
|--- ResidualConnection
|       |--- Initialization
|       |--- Forward Calculation
|--- EncoderBlock
|       |--- Self-Attention Block
|       |--- FeedForward Block
|       |--- Residual Connections
|--- Encoder
|       |--- Multiple Encoder Blocks
|       |--- LayerNormalization
|--- DecoderBlock
|       |--- Self-Attention Block
|       |--- Cross-Attention Block
|       |--- FeedForward Block
|       |--- Residual Connections
|--- Decoder
|       |--- Multiple Decoder Blocks
|       |--- LayerNormalization
|--- ProjectionLayer
|       |--- Initialization
|       |--- Forward Calculation
|--- Transformer
|       |--- Encoder
|       |       |--- Encode Source
|       |--- Decoder
|       |       |--- Decode Target
|       |--- Projection
|--- End
~~~

## Model Summary

![image](https://github.com/prarthanats/ERA/assets/32382676/712aa688-b1d5-4d8f-a696-676855bb7c83)

## Implementation and Inference Details
~~~
	Epochs - 10
	Batch Size - 8
	Number of parameters: 75.1 M  
	loss - 4.64
~~~

## Lightening Code Structure

~~~
|--- Initialization
|--- Prepare Data
|--- Configure Optimizers
|--- Define Train DataLoader
|--- Define Validation DataLoader
|--- Split Dataset into Training and Validation Sets
|--- Training Loop
|--- Training Step
|       |--- Forward Pass
|       |--- Loss Calculation
|       |--- Backpropagation
|       |--- Logging Training Loss
|--- Validation Step
|       |--- Set Model to Evaluation Mode
|       |--- Greedy Decoding for Validation
|       |--- Compute Evaluation Metrics (CER, WER, BLEU)
|       |--- Log Validation Metrics
|--- End of Epoch
|       |--- Log Epoch Number
|       |--- Save Model (if Preloading)

~~~

## Training Loss Plot

<img width="597" alt="training loss" src="https://github.com/prarthanats/ERA/assets/32382676/515a0b86-5e0e-42fa-a8e0-3af71e0355d4">



