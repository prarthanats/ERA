
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
|--- InputEmbeddings
|       |--- nn.Embedding
|--- PositionalEncoding
|       |--- Positional Encoding Calculation
|--- EncoderBlock
|       |--- Self-Attention Block
|       |--- FeedForward Block
|       |--- Residual Connections
|--- DecoderBlock
|       |--- Self-Attention Block
|       |--- Cross-Attention Block
|       |--- FeedForward Block
|       |--- Residual Connections
|--- Encoder
|       |--- Encoder Blocks
|       |--- LayerNormalization
|--- Decoder
|       |--- Decoder Blocks
|       |--- LayerNormalization
|--- Transformer
|       |--- Encoder
|       |       |--- Encode Source
|       |--- Decoder
|       |       |--- Decode Target
|       |--- Projection
~~~

## Model Summary

![image](https://github.com/prarthanats/ERA/assets/32382676/712aa688-b1d5-4d8f-a696-676855bb7c83)

## Implementation and Inference Details
~~~
	Epochs - 10
	Batch Size - 8
	Number of parameters: 75.1 M  
	loss - 3.4823
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

<img width="555" alt="training_loss_new" src="https://github.com/prarthanats/ERA/assets/32382676/4455178c-4efe-4050-8392-f75c22dbad23">

## Validation Character Error Rate

The Character Error Rate (CER) is a metric used to evaluate the accuracy of character-level text recognition. It measures the number of character-level errors made by a system when compared to a reference or ground truth.

<img width="527" alt="cer" src="https://github.com/prarthanats/ERA/assets/32382676/9b4d0df0-c9b4-4e9b-8018-9d78447bbc94">

## Validation Word Error Rate

 The Word Error Rate (WER) is a metric used to measures the number of word-level errors made by a system when compared to a reference or ground truth.

 <img width="545" alt="wer" src="https://github.com/prarthanats/ERA/assets/32382676/105cb47c-65eb-4a39-81a2-37177efdc7df">

## Training Logs
~~~
	Training: 0it [00:00, ?it/s]
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: la mia cosa , e la mia cosa , e la mia cosa .
	Epoch Number 0
	Final Training Loss: 5.8139
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: La mia vita era un ' altra , e mi parve che mi .
	Epoch Number 1
	Final Training Loss: 5.8436
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: La mia vita era stata in una volta , e mi di , e mi di .
	Epoch Number 2
	Final Training Loss: 3.7605
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: la mia testa , che era la notte , e la mia casa mi aveva una volta una volta a me .
	Epoch Number 3
	Final Training Loss: 5.1322
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: La mia meraviglia era stata , e la notte mi di nuovo il mio spirito .
	Epoch Number 4
	Final Training Loss: 5.4552
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: Un ' altra volta , che cosa mi aveva fatto il mio spirito , e la mia solitudine mi pareva che la mia vita mi .
	Epoch Number 5
	Final Training Loss: 4.9846
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: " Quando la notte era svanita , la notte , la notte e la notte mi di coraggio .
	Epoch Number 6
	Final Training Loss: 4.5593
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: Una notte seduta al mio letto , la notte e la notte , la mia solitudine .
	Epoch Number 7
	Final Training Loss: 3.2663
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: Un ' ora , che cosa era in quella notte , la mia solitudine e la mia solitudine .
	Epoch Number 8
	Final Training Loss: 4.2449
	Validation: 0it [00:00, ?it/s]
	    SOURCE: A form was near -- what form , the pitch - dark night and my enfeebled vision prevented me from .
	    TARGET: Una forma mi era vicina, ma come fosse non potevo distinguerla nel buio.
	 PREDICTED: Una sera era in mezzo alla mia notte , che mi sentii la notte e la potenza di veder la potenza e la potenza di .
	Epoch Number 9
	Final Training Loss: 3.4823
~~~


