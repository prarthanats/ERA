
# BERT, GPT and ViT Code Modularity

This repository contains code and utilities for training and evaluating deep learning models, possibly for tasks like machine translation or other sequence-to-sequence tasks using the PyTorch framework.

## Objective
To have a single transformer.py file that we can use to train all 3 models - BERT, GPT and ViT

## Introduction to BERT

BERT, short for Bidirectional Encoder Representations from Transformers, is a revolutionary deep learning model in the field of natural language processing (NLP). Introduced by Google AI in 2018, BERT has significantly impacted various NLP tasks, such as text classification and language generation. This brief overview provides insight into BERT's architecture, components, and its profound significance in NLP.

### BERT Architecture:
BERT is built upon the Transformer architecture, designed initially for sequence-to-sequence tasks. BERT's innovation lies in its bidirectional context understanding, distinguishing it from previous unidirectional models.

Key Components:

Transformer Encoder: BERT employs the Transformer's encoder stack, featuring multiple layers of self-attention mechanisms and feedforward neural networks. These layers enable BERT to capture contextual information from input text effectively.

Bidirectional Training: BERT is pretrained on extensive text data using two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). MLM involves predicting masked words in a sentence, while NSP determines if a random sentence follows another in a document.

![image](https://github.com/prarthanats/ERA/assets/32382676/c2dcfddd-745f-4e51-8c02-55fccc0b1401)


### Significance:
Since its inception, BERT has transformed NLP and inspired subsequent architectures like GPT-3 and T5. BERT-based models underpin various NLP applications, from chatbots to content recommendations. Ongoing research seeks to enhance BERT's efficiency, multilingual support, and fine-tuning techniques, ensuring its enduring influence on NLP.

In conclusion, BERT represents a major breakthrough in natural language understanding, with its bidirectional context awareness and transfer learning capabilities reshaping the NLP landscape.

## Introduction to GPT

GPT, short for "Generative Pre-trained Transformer," comprises a series of advanced deep learning models developed by OpenAI. This series began with GPT-1 in 2018 and progressed to GPT-3 in 2020, exerting a profound influence on the field of natural language processing (NLP). These models have established new benchmarks for language generation, comprehension, and numerous NLP tasks. This concise overview outlines GPT's key characteristics and its significance in the domain of NLP.

### GPT Architecture:
At its core, GPT employs the Transformer architecture, initially designed for sequence-to-sequence tasks. GPT models are tailored for working with text data and consist of fundamental components:

Transformer Decoder: Unlike the original Transformer, GPT exclusively uses the decoder component, featuring multiple layers of self-attention mechanisms and feedforward neural networks.

Pre-training: GPT models are pre-trained on extensive internet text data, learning to predict the next word in a sentence based on preceding context. This process equips GPT with a broad understanding of language, grammar, and world knowledge.

Generative Abilities: A standout feature of GPT models is their capacity for generating coherent and contextually relevant text. This ability makes them versatile for tasks such as text completion, text generation, and even creative writing.

![image](https://github.com/prarthanats/ERA/assets/32382676/646f7d42-5042-41f9-89f2-57c8fa79150f)

### Significance:
The introduction of GPT models has transformed multiple industries, including healthcare, customer service, and content generation. However, it has also raised concerns regarding ethical AI development and responsible AI usage due to the potential for generating realistic yet potentially biased or harmful content.

## Introduction to VIT

ViT, or Vision Transformer, introduced in 2020, is a revolutionary deep learning architecture that has transformed computer vision tasks. Departing from the conventional dominance of Convolutional Neural Networks (CNNs) in image processing, ViT leverages the Transformer architecture originally designed for natural language processing.

### Key Components:

Patch Embedding: ViT divides input images into non-overlapping patches and linearly embeds them into lower-dimensional vectors, serving as model input.
Positional Embeddings: To encode spatial information, ViT introduces positional embeddings added to patch embeddings, facilitating relative patch location understanding.
Transformer Encoder: ViT's core utilizes a Transformer encoder, processing patch and positional embeddings with self-attention mechanisms and feedforward neural networks.
Classification Head: Following patch processing, ViT typically employs a classification head with fully connected layers for predictions, such as image classification.

![image](https://github.com/prarthanats/ERA/assets/32382676/5e7b8ee8-f9d2-4fa3-9b34-a3e8b16ba1ab)

### Significance
ViT has gained widespread attention in the computer vision community, pushing the boundaries of image understanding. Ongoing research explores hybrid models, combining ViT with other architectures, and addresses challenges like efficient high-resolution image handling.

## Code Working

The transformer file includes a class called Transformer -[Transformer](https://github.com/prarthanats/ERA/blob/main/Assignment_17/transformer.py) 

Represents a deep learning model architecture for natural language processing tasks. The class has three main branches for different model types: 'bert,' 'gpt,' and 'vit' (Vision Transformer). Here's a brief overview of each branch:

1. 'bert' Branch:
This branch is designed for BERT (Bidirectional Encoder Representations from Transformers)-style models.
It includes an embedding layer, positional embeddings, multiple encoder layers, layer normalization, and a linear layer for output.
The forward method processes input data through these components and returns the model's output.

2. 'gpt' Branch:
This branch is designed for GPT (Generative Pretrained Transformer)-style models.
It includes token and positional embedding layers, multiple transformer blocks, layer normalization, and a linear head for language modeling.
The forward method processes input data through these components and returns model logits. Optionally, it computes the cross-entropy loss if targets are provided.

3. 'vit' Branch:
This branch is designed for Vision Transformer models.
It assumes the existence of a 'ViT' model and forwards input data through it.

## Training Logs

### BERT

![image](https://github.com/prarthanats/ERA/assets/32382676/d645a001-5117-4d51-885e-f36ac7c40714)

### GPT

![image](https://github.com/prarthanats/ERA/assets/32382676/e8ed6792-4643-4ca5-85bb-0fc35bb1b122)

### VIT

![image](https://github.com/prarthanats/ERA/assets/32382676/c37fa4f8-7b28-49ea-bd24-a4202e403716)


