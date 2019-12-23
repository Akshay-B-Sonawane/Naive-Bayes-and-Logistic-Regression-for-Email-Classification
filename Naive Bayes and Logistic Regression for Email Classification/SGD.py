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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report





#train_data = pd.read_csv("Dataset1_train.csv")
#test = pd.read_csv("Dataset1_test.csv")



#X = train_data.iloc[:, 1:]
#X = np.c_[np.ones((X.shape[0],1)), X]
#y = train_data.iloc[:,0]
#y = y[:, np.newaxis]
#
#X_t = test.iloc[:, 1:]
#X_t = np.c_[np.ones((X_t.shape[0],1)), X_t]
#y_t = test.iloc[:,0]
#y_t = y_t[:, np.newaxis]
#
#from sklearn.model_selection import train_test_split
#X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
#
#
#
#from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import GridSearchCV


#n_iters = [5, 10, 20, 50, 100, 1000]
#scores = []
#for n_iter in n_iters:
#    SGDmodel = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)
#    SGDmodel.fit(X_train, y_train)
#    scores.append(SGDmodel.score(X_test, y_test))
#  
#plt.title("Effect of n_iter")
#plt.xlabel("n_iter")
#plt.ylabel("score")
#plt.plot(n_iters, scores)
#
#
#
#
#
#
#

#params = {
#    "loss" : ["hinge", "log","squared_hinge", "perceptron"],
#    "alpha" : [ 0.001, 0.01, 0.1],
#    "penalty" : ["l2", "l1"],
#}

#params = {
#    "loss" : ["log"],
#    "alpha" : [0.01],
#    "penalty" : ["l2"],
#}
#
#SGDmodel = SGDClassifier( max_iter = 60)
#clf = GridSearchCV(SGDmodel, param_grid=params)
#clf.fit(X_t, y_t)
#print(clf.best_score_)
#
#
#def generateConfusionMatrix(actual, predicted):    
#    results = confusion_matrix(actual, predicted)
#    print('Confusion Matrix :')
#    print(results)
#    print('Accuracy Score :',accuracy_score(actual, predicted))
#    print('Report : ')
#    print(classification_report(actual, predicted) )
#
#generateConfusionMatrix(actual = test.iloc[:,1], predicted = pred)



if __name__ == "__main__":
    

    train = glob.glob(str(sys.argv[1]+"_train.csv"))
    d = str(sys.argv[1])
    itr = int(sys.argv[2])
    train_data = pd.read_csv(train[0])
    test = pd.read_csv(d+"_test.csv")
#    enron = str(sys.argv[2])
#    print(train_data)
    
#    ll = len(train_data)
    X = train_data.iloc[:, 1:]
    X = np.c_[np.ones((X.shape[0],1)), X]
    y = train_data.iloc[:,0]
    y = y[:, np.newaxis]
    
    X_t = test.iloc[:, 1:]
    X_t = np.c_[np.ones((X_t.shape[0],1)), X_t]
    y_t = test.iloc[:,0]
    y_t = y_t[:, np.newaxis]
    
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    
#    n_iters = [5, 10, 20, 50, 100, 1000]
#    scores = []
#    for n_iter in n_iters:
#        SGDmodel = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)
#        SGDmodel.fit(X_train, y_train)
#        scores.append(SGDmodel.score(X_test, y_test))
#      
#    plt.title("Effect of n_iter")
#    plt.xlabel("n_iter")
#    plt.ylabel("score")
#    plt.plot(n_iters, scores)
    
    
    
    
 
    
#    iteration(X_train, y_train, X_test, y_test)
#    score(X_t, y_t, itr)
    
    
    params = {
        "loss" : ["hinge", "log","squared_hinge", "perceptron"],
        "alpha" : [ 0.001, 0.01, 0.1],
        "penalty" : ["l2", "l1"],
    }
    
    SGDmodel = SGDClassifier(max_iter=itr)
    clf = GridSearchCV(SGDmodel, param_grid=params)
    clf.fit(X_t,y_t)
    print(clf.best_score_)
