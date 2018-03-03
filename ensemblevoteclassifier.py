#imports
import re
import pickle
import string
import math
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

nltk.download()
ps = PorterStemmer()
dictionary = []


#save output
def save_list(result, filepath):
    with open(filepath, 'w') as out:
        for x in result:
            out.write(str(x) + '\n')


#pre-process files
def extract(folderpath):
    docs = glob(folderpath)    
    for d in docs:
        file = open(d,'r', errors='ignore')
        readfile = file.readlines()
        readfile = ' '.join(readfile)
        #filter text to remove stopwords, punctuation, numbers
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(readfile)
        punctuation = set(string.punctuation)
        filtered_sentence = ' '.join(w for w in word_tokens if not w in stop_words)
        filtered_sentence = ''.join(ch for ch in filtered_sentence if ch not in punctuation)
        filtered_sentence = ''.join([i for i in filtered_sentence if not i.isdigit()])
        filtered_sentence = filtered_sentence.lower()
        #unigram
        unigram = word_tokenize(filtered_sentence)
        #stemming
        stemming = []
        for u in unigram:
            stemming = stemming+[(ps.stem(u))]
            stemmed = ' '.join(stemming)
        #append into dictionary
        dictionary.append(stemmed)
    return(dictionary)


#train models
def classifier(model):
    model.fit(train_matrix, train_review.Label)
    return(model.predict(test_matrix))


#split train to multiple files and store in folder
train_review = pd.read_csv("C:/Users/aksha/Desktop/Text Analytics/Homework3/train.csv",header=None)
train_review.columns= ['Text', 'Label']
for r in train_review.iterrows():
    moviereview_train = ((r[1]['Text']))
    file = open('C:/Users/aksha/Desktop/Text Analytics/Homework3/Split train/{r}.txt'.format(r= r[0]),'w')
    file.write(moviereview_train)
    file.close()


#read train files
train_dictionary = extract('C:/Users/aksha/Desktop/Text Analytics/Homework3/Split train/*.txt')


#tfidf matrix -- train
tfidf = TfidfVectorizer(min_df = 0.01, max_df = 0.99)
train_matrix = np.asarray(tfidf.fit_transform(train_dictionary).todense())


#split test to multiple files and store in folder
test_review = pd.read_csv("C:/Users/aksha/Desktop/Text Analytics/Homework3/test.csv",header=None)
test_review.columns= ['Text', 'Label']
for r in test_review.iterrows():
    moviereview_test = ((r[1]['Text']))
    file = open('C:/Users/aksha/Desktop/Text Analytics/Homework3/Split test/{r}.txt'.format(r= r[0]),'w')
    file.write(moviereview_test)
    file.close()


#read test files
test_dictionary = extract('C:/Users/aksha/Desktop/Text Analytics/Homework3/Split test/*.txt')


#tfidf matrix -- test
test_matrix = np.asarray(tfidf.transform(test_dictionary).todense())


#Logistic Regression
lr = LogisticRegression(random_state = 1)
logistic = classifier(lr)


#Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state = 1)
tree = classifier(dtc)


#Random Forest Classifier
rf = RandomForestClassifier(random_state = 1)
rfc = classifier(rf)


#Voting CLassifier
voting = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('tree', dtc)], voting='hard')
voting = classifier(voting)


#save files
save_list(logistic, 'C:/Users/aksha/Desktop/Text Analytics/Homework3/Output/classification_logistic.txt')
save_list(tree, 'C:/Users/aksha/Desktop/Text Analytics/Homework3/Output/classification_tree.txt')
save_list(rfc, 'C:/Users/aksha/Desktop/Text Analytics/Homework3/Output/classification_rf.txt')
save_list(voting, 'C:/Users/aksha/Desktop/Text Analytics/Homework3/Output/classification_voting.txt')

