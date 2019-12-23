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
import math
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#for i in glob.glob("./train/*"):a

def func(path, cl):
    i= path
    c = -1
    vocabulary = set()
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
            stem_sen = []
            for word in results:
                stem_sen.append(stemmer.stem(word))
            for word in stem_sen:
                if word not in train.columns:
                    train[word] = [0]*len_folder
                train.loc[c,word] +=1
#                print(c,word)
            stem_sen = set(stem_sen)
#            for word in stem_sen:
#                if word not in train.columns:
#                    train_bernuli[word] = [0]*len_folder
#                train_bernuli.loc[c,word] +=1
#                print(c,word)
            
            vocabulary = vocabulary.union(stem_sen)
        
    
    return train
ham = func(path = "./train/ham",cl = 1)
spam = func(path = "./train/spam",cl = 0)

train_data = ham.append(spam).fillna(0)
train_data.index = range(len(train_data.index))


train_data.to_csv(r'F:\UTD\Machine Learning\Dataset1\bog.csv', index=False)

_,c = train_data.shape
label = train_data.iloc[:,0]
data = train_data.iloc[:,1:c]

#train_data = train_data.fillna(0)
#vocabulary = set(vocabulary)

    
#def bagofwords(sentence, words):
#    sentence_words = sentence
#    # frequency word count
#    bag = np.zeros(len(words))
#    for sw in sentence_words:
#        for i,word in enumerate(words):
#            if word == sw: 
#                bag[i] += 1
#    return np.array(bag)
#            
#bag = bagofwords(stem_sen, tokens)



def prior(train):
    train_ham = train[train["Label"] == 1]
    train_spam = train[train["Label"] == 0]
    prior_ham = len(train_ham)/len(train)
    prior_spam = len(train_spam)/len(train)
    return prior_ham, prior_spam

p_ham, p_spam = prior(train_data)

def condProb(train):
    _,c = train.shape
    train_ham = train[train["Label"] == 1]
    train_spam = train[train["Label"] == 0]
    num_ham = (train_ham.iloc[:,1:c].sum(axis = 0)) + 1
    den_ham = (c-1) + (train_ham.iloc[:,1:c].sum(axis = 0)).sum()
    
    num_spam = (train_spam.iloc[:,1:c].sum(axis = 0)) + 1
    den_spam = (c-1) + (train_spam.iloc[:,1:c].sum(axis = 0)).sum()
    dic_ham = (num_ham/den_ham)
    dic_spam = (num_spam/den_spam)
    return dic_ham, dic_spam, den_ham, den_spam

dic_ham, dic_spam, den_ham, den_spam = condProb(train_data)


def func(path, classification):
    i= path
    c = -1
#    vocabulary = set()

#    train_bernuli = pd.DataFrame()
#    len_folder = len(glob.glob(i+"/*.txt"))
#
#    
#    train['Label'] = [0]*len_folder
    row = []
    index = []
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
            stem_sen = []
            for word in results:
                stem_sen.append(stemmer.stem(word))
            row.append({'Message': stem_sen, 'class': classification})
            index.append(c)
    
    return DataFrame(row, index=index)
    
test_ham = DataFrame({'Message': [], 'class': []})
test_ham = func(path = "./test/ham", classification = 1)

test_spam = DataFrame({'Message': [], 'class': []})
test_spam = func(path = "./test/spam", classification = 0)

test = test_ham.append(test_spam)
test.index = range(len(test.index))



def APPLYMULTINOMIALNB(p_ham, p_spam, dic_ham, dic_spam, data, vocab, den_ham, den_spam):
#    counter = 0
    classification = np.array([0]*len(data))
    for i in range(len(data)):
        d = data.iloc[i,:]
        W = d[0]
#        W = vocab.intersection(W)
        score_ham = np.log(p_ham)
        score_spam = np.log(p_spam)
        for t in W:
            try:
                score_ham += np.log(dic_ham[t]) 
                score_spam += np.log(dic_spam[t])
            except:
                score_ham += np.log(1/(den_ham))
                score_spam += np.log(1/(den_spam))
            
        score = np.array([score_spam,score_ham])
        classification[i] = score.argmax()
#        classification = score.argmax()
#        if classification == d[1]:
#            counter += 1
#    return counter
    return classification
    
def generateConfusionMatrix(actual, predicted):    
    results = confusion_matrix(actual, predicted)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :',accuracy_score(actual, predicted))
    print('Report : ')
    print(classification_report(actual, predicted) )

vocab = list(train_data.columns)
Test = APPLYMULTINOMIALNB(p_ham, p_spam, dic_ham, dic_spam, test, vocab, den_ham, den_spam)

generateConfusionMatrix(actual = test.iloc[:,1], predicted = Test)



#d = test.iloc[2,:]
#W = set(d[0])
#W = vocab.intersection(W)
#score_ham = np.log(p_ham)
#score_spam = np.log(p_spam)
#for t in W:
#    score_ham += np.log(dic_ham[t]) 
#    score_spam += np.log(dic_spam[t])
#    



#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(stem_sen).toarray()