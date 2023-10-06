
# VAE on MNIST and CIFAR10

This repository contains code and utilities implementation of VAE on the MNIST and CIFAR10 datasets.

## Introduction to Variational Autoencoders

Variational Autoencoders (VAEs) are a type of generative model used in machine learning for unsupervised learning and data generation tasks. Here's a brief explanation of VAEs:

### Encoder-Decoder Architecture:

VAEs consist of two main components: an encoder and a decoder.
The encoder takes input data and maps it to a latent space, where each point in the latent space represents a compressed and encoded version of the input data.
The decoder then takes points from the latent space and reconstructs the original data from them.

### Variational Inference:

VAEs use a probabilistic approach for encoding data into the latent space. Instead of producing a single fixed point in the latent space for each input, VAEs generate a probability distribution over the latent space.
This probabilistic aspect allows VAEs to capture uncertainty and generate diverse samples during the decoding process.
Latent Space and Sampling:

The latent space serves as a continuous, lower-dimensional representation of the data.
To generate new data, VAEs sample points from the latent space distribution and pass them through the decoder.
This stochastic sampling process enables the generation of new, similar data samples that share characteristics with the original data.

### Objective Function - Variational Lower Bound:

VAEs are trained using a loss function called the Evidence Lower Bound (ELBO), which encourages the encoder to map data points into a latent space that follows a specific probability distribution (typically a Gaussian distribution).
The ELBO consists of two terms: a reconstruction term that measures how well the decoder can reconstruct the input data from the latent space, and a regularization term that encourages the latent space distribution to be close to a predefined distribution (usually a standard Gaussian).
Applications:

VAEs find applications in various domains, including image generation, data denoising, representation learning, and more.
They are often used for generating novel data samples, interpolating between existing data points, and discovering meaningful representations of complex data.

![image](https://github.com/prarthanats/ERA/assets/32382676/7c3bbf9b-6f02-4f74-9b6c-cac3c97a8909)

## VAE on MNIST


