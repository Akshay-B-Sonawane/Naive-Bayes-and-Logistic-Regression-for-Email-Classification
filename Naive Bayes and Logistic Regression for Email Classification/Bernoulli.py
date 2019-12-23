import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pandas import DataFrame
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import re
import pandas as pd
import glob
import sys
import math
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



#for i in glob.glob("./train/*"):

def func(path, cl):
    i= path
    c = -1
#    vocabulary = set()
    train = pd.DataFrame()
#    train_bernuli = pd.DataFrame()
    len_folder = len(glob.glob(i+"/*.txt"))

    
    train['Label'] = [cl]*len_folder
    for j in glob.glob(i+"/*.txt"):
        c+=1 
        print(c,j)
        with open(j,encoding = 'latin-1') as f:
            file = f.read()
            file = file.lower()
            file = re.sub(r'\d+', "", file)
            file = re.sub(r'[^\w\s]','',file)
            file = " ".join(file.split())
            tokens = word_tokenize(file)
            stop_words = set(stopwords.words('english'))
            results = [i for i in tokens if not i in stop_words]
            stem_sen = set()
            for word in results:
                try:
                    stem_sen.add(stemmer.stem(word))
                except:
                    IndexError
#            for word in results:
#                stem_sen.append(stemmer.stem(word))
            for word in stem_sen:
                if word not in train.columns:
                    train[word] = [0]*len_folder
                train.loc[c,word] +=1
#                print(c,word)
#            stem_sen = set(stem_sen)
##            for word in stem_sen:
##                if word not in train.columns:
##                    train_bernuli[word] = [0]*len_folder
##                train_bernuli.loc[c,word] +=1
##                print(c,word)
#            
#            vocabulary = vocabulary.union(stem_sen)
#        
    
    return train



if __name__ == "__main__":

    train = glob.glob(str(sys.argv[1]+"_train.csv"))
    d = str(sys.argv[1])
    enron = str(sys.argv[2])

    ham  = func(path = "./"+d+"/"+enron+"/train/ham", cl = 1)
    spam = func(path = "./"+d+"/"+enron+"/train/spam",cl = 0)
    
    train_data = ham.append(spam).fillna(0)
    train_data.index = range(len(train_data.index))
    print(train_data.head())
    
    
    t_ham = func(path = "./"+d+"/"+enron+"/test/ham", cl = 1)
    t_spam = func(path = "./"+d+"/"+enron+"/test/spam",cl = 0)
    
    test = t_ham.append(t_spam).fillna(0)
    test.index = range(len(test.index))
    print(test.head())