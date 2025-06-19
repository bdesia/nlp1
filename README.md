# NLP1
CEIA FIUBA - Natural Language Processing 1

This repository contains different challenges solved during the course.

Author: Braian Desía (b.desia@hotmail.com)

**Challenge #1: Text Vectorization and Classification**

This project demonstrates how to vectorize text documents and perform classification using various NLP techniques. It covers the entire from feature extraction to model evaluation and hyperparameter tuning.

*Main features*
- TF-IDF vectorization of documents to convert raw text into numerical features.
- Cosine similarity evaluation to measure document similarity.
- Implementation of Multinomial Naive Bayes and Complement Naive Bayes classifiers using TF-IDF features for document classification.
- Hyperparameter tuning with GridSearchCV to optimize model performance.

*Notebook:* [Desafio_1.ipynb](Desafio_1.ipynb)

**Challenge #2: Custom embeddings with Gensim**

This project is focus on generating embeddings using Word2Vec model from Gensim and how to visualize them. Word similarity is explored.

*Main features*
- Utilizes the "Coronavirus Tweets NLP - Text Classification" dataset obtained from Kaggle.
- Applies Word2Vec from Gensim to generate embeddings.
- Uses cosine similarity to evaluate similarity between words.
- Visualizes embeddings in 2D using both PCA and t-SNE techniques.

*Notebook:* [Desafio_2.ipynb](Desafio_2.ipynb)

**Challenge #3: LSTM Next Word Prediction Project**

This project demonstrates how to build a next word prediction model using a Long Short-Term Memory (LSTM) network. The model is trained on Shakespeare's *Hamlet* from the NLTK Gutenberg corpus and learns to predict the next word in a sequence.

*Features*
- Utilizes *Hamlet* from the NLTK Gutenberg corpus as training data.
- Tokenization and n-gram sequence generation for preparing training data.
- Model #1: Unidirectional LSTM network for basic next word prediction.
- Model #2: Bidirectional LSTM network for better context understanding.
- Ability to **predict the next word** given a sequence.
- Save and load trained models and tokenizers for future use.

*Notebook:* [Desafio3/Desafio_3.ipynb](Desafio3/Desafio_3.ipynb)


**Challenge #4: QA bot Project**

This project implies a Question-Answering (QA) Bot using sequence-to-sequence models with LSTMs, trained on conversational data.

*Features*
- Utilizes data from Conversational Intelligence Challenge 2
- Tokenization and n-gram sequence generation for preparing training data.
- Use of FastText pre-trained embeddings.
- LSTM-based encoder-decoder architecture for Seq2Seq model.
- Ability to **answer** given a question.
- Save and load trained models and encoder/decoders for future use.

*Notebook:* [Desafio4/Desafio_4.ipynb](Desafio4/Desafio_4.ipynb)
