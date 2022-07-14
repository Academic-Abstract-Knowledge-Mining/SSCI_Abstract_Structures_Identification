import csv
import os
import re
# import nltk
import scipy
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
import sys
import pickle


def get_data(filename):
    df = pd.read_csv(filename, sep='\t', header=None, quoting=3)
    df = df.dropna(axis=0, how='any')
    y = df[0].values
    x = df[1].values
    print(x.shape)
    return x, y

def predict(fold):
    test_file = "Abst{}/test.tsv".format(fold)
    X_test, y_test = get_data(test_file)
    model_path = "model_{}.pkl".format(fold)
    print("restore model...")
    with open(model_path, "rb") as fr:
        vec_clf = pickle.load(fr)
    print("loaded model. start predict")
    y_pred = vec_clf.predict(X_test)
    print(sklearn.metrics.classification_report(y_test,y_pred, digits=4))  
    print(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))

# SVM classifier
def svmClassifier(X_train,y_train):
        vec = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf = True,use_idf = True, ngram_range=(1, 2))
        svm_clf = svm.SVC(C=2.0, kernel='rbf', gamma=0.5, max_iter=10000, cache_size=4000, verbose=True, random_state=42)
        # svm_clf = svm.LinearSVC(C=0.04, max_iter=1000)
        vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
        vec_clf.fit(X_train,y_train)
        return vec_clf
        
# NB classifier
def NBClassifier(X_train,y_train):
        vec = TfidfVectorizer(min_df=3, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 1))
        nb_clf = MultinomialNB()
        vec_clf = Pipeline([('vectorizer', vec), ('pac', nb_clf)])
        vec_clf.fit(X_train,y_train)
        return vec_clf

# train function
def train(arg, fold):
        # X_train, X_test, y_train, y_test = getTrainingAndTestData()
        train_file = "Abst{}/train.tsv".format(fold)
        test_file = "Abst{}/test.tsv".format(fold)
        print(train_file)
        print(test_file)
        X_train, y_train = get_data(train_file)
        X_test, y_test = get_data(test_file)

        if arg=='nb':
            vec_clf = NBClassifier(X_train, y_train)
        elif arg=='svm':
            vec_clf = svmClassifier(X_train, y_train)
        with open("model_{}.pkl".format(fold), "wb") as fw:
            pickle.dump(vec_clf, fw)

        y_pred = vec_clf.predict(X_test)
        print(sklearn.metrics.classification_report(y_test,y_pred, digits=4))  
        print(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        
if __name__ == "__main__":
    model="svm"
    fold=sys.argv[1]
    mode=sys.argv[2]
    if mode == "train":
        train(arg=model, fold=fold)
    elif mode == "test":
        predict(fold)
        

