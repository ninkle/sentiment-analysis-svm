import os
import time
import glob
import re
import nltk
from nltk.corpus import names

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import classification_report

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    neg_data = joblib.load('neg_data.pkl')
    pos_data = joblib.load('pos_data.pkl')
    neutral_data = joblib.load('neutral_data.pkl')

    neg_labels = ['neg'] * len(neg_data)
    pos_labels = ['pos'] * len(pos_data)
    neutral_labels = ['neutral'] * len(neutral_data)

    # split features and labels
    train_size_neg = int(len(neg_data)*0.8)
    train_size_pos = int(len(pos_data)*0.8)
    train_size_neutral = int(len(neutral_data)*0.8)

    train_data = neg_data[:train_size_neg] + pos_data[:train_size_pos] + neutral_data[:train_size_neutral]
    train_labels = neg_labels[:train_size_neg] + pos_labels[:train_size_pos] + neutral_labels[:train_size_neutral]

    test_data = neg_data[train_size_neg:] + pos_data[train_size_pos:] + neutral_data[train_size_neutral:]
    test_labels = neg_labels[train_size_neg:] + pos_labels[train_size_pos:] + neutral_labels[train_size_neutral:]

    # use all data
    # train_data = neg_data + pos_data + neutral_data
    # train_labels = neg_labels + pos_labels + neutral_labels

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    # text -> feature vectors
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # classification with Linear SVC
    lin = svm.LinearSVC()
    t0 = time.time()
    lin.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_lin = lin.predict(test_vectors)
    t2 = time.time()
    time_lin_train = t1 - t0
    time_lin_predict = t2 - t1

    # classification report
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_lin_train, time_lin_predict))
    print(classification_report(test_labels, prediction_lin))

    saved_vectorizer = joblib.dump(vectorizer, 'vectorizer.pkl')
    saved_classifier = joblib.dump(lin, 'classifier.pkl')