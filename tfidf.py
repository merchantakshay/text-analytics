import re
import pickle
import string
import math
import numpy
from glob import glob
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


#save output
def save_result(obj, filepath):
    with open(filepath, 'wb') as out:
        pickle.dump(obj, out)

        
#read files
folderpath='C:/Users/aksha/Desktop/Text Analytics/Homework1/univ/*.txt'
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
import numpy
tfidf = TfidfVectorizer()
matrix = numpy.asarray(tfidf.fit_transform(dictionary).todense())


#cosine similarities
def cosine_similarity(a, b):
    xy, xx, yy = 0,0,0
    for i in range(len(a)):
        x = a[i]; y = b[i]
        xx += x*x
        yy += y*y
        xy += x*y
    return(xy/math.sqrt(xx*yy))        


def pairwise_cosine_similarities(matrix):
    return {(i, j): cosine_similarity(matrix[i], matrix[j])
        for i in range(matrix.shape[1]) for j in range(matrix.shape[1])} 


UIN = '111'
save_result(matrix, UIN + '_matrix.pkl')
save_result(pairwise_cosine_similarities, UIN + '_similarity.pkl')

