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






def func(path = "./test/spam", c = -1, classification=0):
    i= path
#    c = -1
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
#            tokens = word_tokenize(file)
#            stop_words = set(stopwords.words('english'))
#            results = [i for i in tokens if not i in stop_words]
#            stem_sen = []
#            for word in results:
#                stem_sen.append(stemmer.stem(word))
            row.append({'Message': file, 'class': classification})
            index.append(c)
    
    return DataFrame(row, index=index)
    
test_spam = DataFrame({'Message': "", 'class': []})

test_spam = func()

test = test_ham.append(test_spam)
test.index = range(len(test.index))


last = train.iloc[:,-1]
last = last[:,np.newaxis]
yee["Label"] = last
d = train.iloc[:,0]

count = CountVectorizer()
bag = count.fit_transform(d)
features = count.get_feature_names()
yee = pd.DataFrame(bag.toarray(), columns = features)
    
    
    
    
    
    
    
    
    
    
    
def prior(train):
    train_ham = train[train["Label"] == 1]
    train_spam = train[train["Label"] == 0]
    prior_ham = len(train_ham)/len(train)
    prior_spam = len(train_spam)/len(train)
    return prior_ham, prior_spam

p_ham_1, p_spam_1 = prior(yee)



def condProb(train):
    ham_doc = train[train["Label"] == 1]
    spam_doc = train[train["Label"] == 0]
    num_ham = ham_doc.sum() + 1
    num_spam = spam_doc.sum() + 1
    den_ham  = len(ham_doc) + 2
    den_spam = len(spam_doc) + 2
    dic_ham = num_ham/den_ham
    dic_spam = num_spam/den_spam
    return dic_ham, dic_spam

dic_ham, dic_spam = condProb(yee)










def APPLYBERNOULLINB(p_ham, p_spam, dic_ham, dic_spam, data, vocab):
#    vocab = list(data.columns)
    for i in range(len(data)):
        d = data.iloc[i,:]
        W = d[0]
        score_ham = np.log(p_ham)
        score_spam = np.log(p_spam)
        for t in vocab:
            if t in W:
                score_ham += np.log(dic_ham[t])
                score_spam += np.log(dic_spam[t])
            else:
                score_ham += np.log(1-dic_ham[t])
                score_spam += np.log(1-dic_spam[t])
        score = np.array([score_spam,score_ham])
        classification = np.argmax(score)

    return sum(classification == d[1])

vocab = list(yee.columns)
Test = APPLYBERNOULLINB(p_ham_1, p_spam_1, dic_ham, dic_spam, test, vocab)