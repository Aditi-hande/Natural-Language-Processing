# Natural-Language-Processing
Code, Report and Data of some NLP techniques/Algorithms

1. Multinomial Classification
   The dataset is downloadable at: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz. 
   Used NLTK package to process dataset- remove the stop words- perform lemmatization. 
   Used sklearn to extract TF-IDF features. 
   Trained Perceptron, SVM, Logistic Regression and Multinomial Naive Bayes model

3. HMM POS Tagging using Greedy and Viterbi Algorithms
  Created Vocabulary from train as an input to HMM (vocab.txt). 
  Trained HMM by calculating transition and emission probabilities. 
  Implemented the greedy and viterbi decoding algorithms. 

 4. Word2Vec FNN and RNN
    Generated Word2Vec features for the dataset used Gensim library for this purpose. 
    Load the pretrained “word2vec-google-news-300” Word2Vec model and learned how to extract word embeddings. 
    Trained a Word2Vec model using dataset and compared with “word2vec-googlenews-300” Word2Ve features. 
    Trained a single perceptron and an SVM model for the classification problem. 
    Using the Word2Vec features, trained a feedforward multilayer perceptron network for classification. Consider a network with two hidden layers, each with 100 and 10 nodes. 
    Using the Word2Vec features, trained a recurrent neural network (RNN) by considering a gated recurrent unit cell and an an LSTM unit cell. 

5. DL models on named entity recognition (NER)
   Used the CoNLL-2003 corpus to build a neural network for NER. 
   Built a simple bidirectional LSTM model with the training data on NER with SGD as the optimizer. 
   Used the GloVe word embeddings to improve the BLSTM, equip the BLSTM model in Task 2 with a CNN module to capture character-level information. 
