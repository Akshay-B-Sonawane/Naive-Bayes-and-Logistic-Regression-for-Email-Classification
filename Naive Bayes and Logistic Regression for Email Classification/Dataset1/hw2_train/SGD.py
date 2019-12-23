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
import matplotlib.pyplot as plt


def func(path = "./train/spam", c = -1):
    i= path
#    c = -1
    vocabulary = set()
    train = pd.DataFrame()
#    train_bernuli = pd.DataFrame()
    len_folder = len(glob.glob(i+"/*.txt"))

    
    train['Label'] = [0]*len_folder
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
spam = func()

train_data = ham.append(spam).fillna(0)
train_data.index = range(len(train_data.index))


X = train_data.iloc[:, 1:]
X = np.c_[np.ones((X.shape[0],1)), X]
y = train_data.iloc[:,0]
y = y[:, np.newaxis]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)



from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV



n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    model = SGDClassifier(loss="log", penalty="l2", max_iter=n_iter)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)







params = {
    "loss" : ["log"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2"],
}

model = SGDClassifier(max_iter=1000)
clf = GridSearchCV(model, param_grid=params)
clf.fit(X,y)
print(clf.best_score_)