# Word Prediction Model using Neural Networks and Word Embeddings

This project implements a simple neural network model for word prediction based on the context of a given word in a text corpus. Inspired by the principles of the Word2Vec algorithm, the model learns word embeddings that capture the semantic relationships between words based on their neighboring words in a given text.

## Project Overview

The goal of this project is to build a word prediction model that can predict neighboring words based on a word input. The model learns the contextual relationships of words from an input text by encoding words into one-hot vectors and training a neural network to predict words in proximity. The neural network is trained to create 2-dimensional word embeddings which can later be used to measure distances and similarities between words.

### Key Features

- **One-hot encoding** of input and neighboring words.
- **Neural network architecture** with an embedding layer and softmax output.
- **Training using gradient descent** to learn word embeddings.
- **Prediction of context words** based on distance between embeddings.
- **Customizable window size** around a target word to select context words.

## How It Works

1. **Data Preparation**: 
   - The input text is tokenized into individual words, and unique words are mapped to integers. 
   - Pairs of words within a fixed context window are generated for training (word, context-word).

2. **Model**: 
   - A simple neural network with one hidden layer is used to learn 2-dimensional word embeddings.
   - The network is trained using gradient descent to minimize the error between the predicted and actual neighboring words.

3. **Prediction**:
   - After training, the model can predict the most probable neighbors of a word by calculating the Euclidean distance between embeddings.

## Future Improvements
   - Use larger embedding dimensions for more accurate word predictions.
   - Implement additional optimization techniques like Negative Sampling or Hierarchical Softmax.
   - Train the model on a larger and more diverse corpus of text.
   - Integrate pre-trained embeddings like Word2Vec, GloVe, or FastText.
