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

nltk.download()
ps = PorterStemmer()
dictionary = []


#split to multiple files and store in folder
review = pd.read_csv("C:/Users/aksha/Desktop/Text Analytics/Homework2/movie_reviews.csv",header=None)
review.columns= ['Label', 'Text']
for r in review.iterrows():
    moviereview = ((r[1]['Text']))
    file = open('C:/Users/aksha/Desktop/Text Analytics/Homework2/Split files/{r}.txt'.format(r= r[0]),'w')
    file.write(moviereview)
    file.close()


#read files
folderpath='C:/Users/aksha/Desktop/Text Analytics/Homework2/Split files/*.txt'
docs = glob(folderpath)


#open files
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
    
    #pos tagging
    unigram = word_tokenize(filtered_sentence)
    postagged = nltk.pos_tag(unigram)
    
    #bigram
    bigram = ngrams(unigram,2)
    bigrams = list(bigram)
    
    #stemming
    stemming = []
    for u in unigram:
        stemming = stemming+[(ps.stem(u))]
        stemmed = ' '.join(stemming)
            
    #append into dictionary
    dictionary.append(stemmed)


#tfidf matrix
tfidf = TfidfVectorizer()
matrix = np.asarray(tfidf.fit_transform(dictionary).todense())


#mean square error
def mse(y, yh):
    for i in range(len(y)):
        mse = np.square(y[i]-yh[i]).mean()
    return(mse)           


#optimize mean square error using gradient descent
def mse_diff(y, yh):
    cost = 0
    for i in range(len(y)):
        cost += (y[i]-yh[i])
    return(2*cost/len(y))


#linear function
def linear_regression(X, y, lr, n_epoch):
    array=[]
    theta = np.random.randn()*X.shape[1]
    for i in range(n_epoch):
        yb = np.dot(matrix,theta)
        theta -= (lr)*mse_diff(yb,y)
        print("Iteration:{i}; MSE:{mse}".format(i=i,mse=mse(y,yb)))
        array.append(mse(y,yb))
    return(array)


#plot MSE
def plot(array):
    plt.plot(array)
    plt.xlabel('No of Iterations')
    plt.ylabel('Mean Square Error')
    plt.savefig('C:/Users/aksha/Desktop/Text Analytics/Homework2/error_rate.png')
    plt.show()


array = linear_regression(matrix, review.Label, 0.05, 200)
plot(array)

