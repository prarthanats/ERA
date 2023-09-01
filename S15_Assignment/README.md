
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

### Sample Data Sample

![image](https://github.com/prarthanats/ERA/assets/32382676/bb74083d-c064-4676-8367-84744075a308)

### [Code Structure DataSet]('https://github.com/prarthanats/ERA/blob/main/S15_Assignment/lightening_code/dataset.py')
~~~
	- Retrieves an example from the dataset at the specified index and tokenizes the source and target text using the respective tokenizers.
	- Prepares encoder input, decoder input, and labels as follows:
	- Encoder Input: Includes special tokens [SOS] (start of sentence), [EOS] (end of sentence), and padding tokens. The source text is embedded between [SOS] and [EOS], and padding tokens are added as necessary to reach the seq_len.
	- Decoder Input: Includes [SOS] and padding tokens. The target text is embedded after [SOS], and padding tokens are added to match the seq_len. It does not include [EOS] because the decoder has to predict it.
	- Labels: Include the target text followed by [EOS] and padding tokens to reach the seq_len.
	- Returns a dictionary containing the following:
	"encoder_input": Encoder input sequence.
	"decoder_input": Decoder input sequence.
	"encoder_mask": A mask indicating the positions of non-padding tokens in the encoder input.
	"decoder_mask": A mask used in the decoder for causal masking, preventing it from attending to future tokens.
	"label": The label sequence.
	"tgt_text": The target text in its original form
~~~

## [Custom Transformer Model Architecture / Code Structure]('https://github.com/prarthanats/ERA/blob/main/S15_Assignment/lightening_code/model.py')

Summary of the key components:
    - LayerNormalization: Implements layer normalization with learnable scale and bias parameters.
    - FeedForwardBlock: A feedforward neural network block used within the Transformer layers.
    - InputEmbeddings: Embeds input tokens into a continuous vector space.
    - PositionalEncoding: Adds positional information to the input embeddings to account for token order.
    - ResidualConnection: Adds residual connections and layer normalization to sub-layers within the Transformer blocks.
    - MultiHeadAttentionBlock: Implements multi-head self-attention mechanism within the Transformer.
    - EncoderBlock: A building block for the encoder part of the Transformer.
    - Encoder: Stacks multiple encoder blocks to form the encoder of the Transformer.
    - DecoderBlock: A building block for the decoder part of the Transformer.
    - Decoder: Stacks multiple decoder blocks to form the decoder of the Transformer.
    - ProjectionLayer: The final layer that projects decoder outputs into a target vocabulary space.
    - Transformer: Combines the encoder, decoder, embeddings, positional encodings, and projection layers to create the overall Transformer model.
    - build_transformer: A function for constructing the Transformer model with specified parameters.
    
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



## [Pytorch Lightening Code Structure]('https://github.com/prarthanats/ERA/blob/main/S15_Assignment/lightening_code/lightening_train.py')
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

## Setup and Usage

The model can be run using Lightning framework as shown below:

1. Clone the GitHub Repo
~~~
!git clone https://github.com/prarthanats/ERA/tree/main/S15_Assignment.git
~~~

2. Import Dependency
~~~
	
	from callback import TrainEndCallback
	from lightening_train import PytorchLighteningTransformer
	from config import get_config, get_weights_file_path

	import warnings
	from tqdm import tqdm
	import os
	from pathlib import Path
	
	import torchmetrics
	from torch.utils.tensorboard import SummaryWriter
	
	from pytorch_lightning import LightningModule
	from pytorch_lightning.callbacks.progress import TQDMProgressBar
	
	import pytorch_lightning as pl
	from pytorch_lightning.loggers import TensorBoardLogger

~~~
3. Model Training and Testing

3.1 TrainEndCallback
This save the model's state at the end of each training epoch and prints the epoch and Average Loss

3.2 Instantiating the Trainer
trainer: This is an instance of the PyTorch Lightning Trainer class, which coordinates the training process. It's configured with various options:max_epochs=10: Training will run for a maximum of 10 epochs. callbacks=callbacks: The list of callbacks defined earlier is passed to the Trainer to be used during training.

3.3 Model Training and Testing
The trainer orchestrates the entire training loop, handling data loading, batching, forward and backward passes, optimization, and other training-related tasks.
~~~
	model = PytorchLighteningTransformer(config)
	callback = TrainEndCallback(config=config)
	
	# Create the Trainer instance with the callback
	trainer = pl.Trainer(callbacks=[callback],max_epochs=10)
	
	# Train the Lightning module
	trainer.fit(model)
~~~


## Implementation and Inference Details
~~~
	Epochs - 10
	Batch Size - 8
	Number of parameters: 75.1 M  
	loss - 3.4823
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

We can see that in the 10th epoch the loss is below 4.
 
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


