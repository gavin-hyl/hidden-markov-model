# pythonhmm - Hidden Markov Model Implementation in Python

## Features

Implementation of all major algorithms related to Hidden Markov Models

- Viterbi Algorithm: calculating maximum probability state sequences
- Forward-backward algorithm: calculating joint and conditional probabilities for segments in sequences
- Supervised learning: training a model based on labelled data
- Unsupervised learning (Baum-Welch algorithm): train a model on unlabelled data using a special case of the EM algorithm.

Also provides utilities for data preprocessing for HMMs, such as sequence indexing.

For examples, see ``examples.ipynb``.

## Dependencies

- numpy
