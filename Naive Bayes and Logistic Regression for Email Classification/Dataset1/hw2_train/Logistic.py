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

ham  = func(path = "./train/ham", cl = 1)
spam = func(path = "./train/spam",cl = 0)

train_data = ham.append(spam).fillna(0)
train_data.index = range(len(train_data.index))


t_ham = func(path = "./test/ham", cl = 1)
t_spam = func(path = "./test/spam",cl = 0)

test = t_ham.append(t_spam).fillna(0)
test.index = range(len(test.index))



X = train_data.iloc[:, 1:]
X = np.c_[np.ones((X.shape[0],1)), X]
y = train_data.iloc[:,0]
y = y[:, np.newaxis]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)



def sigmoid(x):
    return 1/(1+np.exp(-x))



def costfunction(X,y, theta, Lambda, alpha):
    m = len(y)
    pred = sigmoid(X @ theta)
    error = (-y*np.log(pred)) - ((1-y)*np.log(1-pred))
#    error = (y*(X @ theta) - np.log(1 + np.exp(X @ theta)))
    
    J = 1/m*sum(error)
    reg_J = J +  (Lambda/(2*m)) * np.sum(theta**2)
#    J_0 = 1/m*(X.transpose() @ (pred - y))[0]
#    J_1 = 1/m*(X.transpose() @ (pred - y))[1:] - alpha*(Lambda/m)*theta[1:]
    grad = (X.transpose() @ (y-pred))
#    grad = np.vstack((J_0[:,np.newaxis],J_1))
    return reg_J[0], grad

#
#theta = np.zeros((X.shape[1],1))
#
#
#cost, grad = costfunction(X,y, theta, Lambda)    
#print("Cost at initial theta (zeros): ", cost)




def gradientAscent(X,y, theta, alpha, num_iteration, Lambda):
    m = len(y)
    J_history = []
    
    for i in range(num_iteration):
        cost, grad = costfunction(X,y,theta, Lambda, alpha)
        theta = theta + (alpha*grad - alpha*Lambda*theta)
#        theta = theta + (alpha*grad)
        J_history.append(cost)
    return theta, J_history
    

theta = np.zeros((X.shape[1],1))
Lambda = 3
    
theta , J_history = gradientAscent(X_train,y_train,theta,0.2, 1000, Lambda)

print("The regularized theta using ridge regression:\n",theta)

for itr in (900, 1000, 1500):
    for alpha in (0.1, 0.20, 0.29, 0.3, 0.01):
        for Lambda in (-3,-2,-1,1,2,3):
            print(" ")
            print("For itr: ", itr, "aplha: ", alpha, "Lambda: ", Lambda)
            theta = np.zeros((X.shape[1],1))
            theta , J_history = gradientAscent(X_train,y_train,theta,alpha, itr, Lambda)
            p=classifierPredict(theta,X_train)
            print("Train Accuracy:", (sum(p==y_train)/len(y_train) *100)[0],"%")
            p=classifierPredict(theta,X_test)
            print("Test Accuracy:", (sum(p==y_test)/len(y_test) *100)[0],"%")
    



def classifierPredict(theta,X):
    """
    take in numpy array of theta and X and predict the class 
    """
    predictions = X.dot(theta)
    
    return predictions>0

p=classifierPredict(theta,X_train)
print("Train Accuracy:", (sum(p==y_train)/len(y_train) *100)[0],"%")
p=classifierPredict(theta,X_test)
print("Test Accuracy:", (sum(p==y_test)/len(y_test) *100)[0],"%")




def func_message(path, classification):
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
test_ham = func_message(path = "./test/ham", classification = 1)

test_spam = DataFrame({'Message': [], 'class': []})
test_spam = func_message(path = "./test/spam", classification = 0)

test_data = test_ham.append(test_spam)
test_data.index = range(len(test_data.index))


train_data.to_csv(r'F:\UTD\Machine Learning\Dataset1\file3.csv', index=False)




def testClassification(data, test, theta, vocabulary):
#    counter = 0
    ped = np.array([0]*len(data))

    for i in range(len(data)):
        d = data.iloc[i,:]
        W = d[0]
        pred = 0
        for t in W:
            if t in vocabulary:
                pred += test.loc[i,t]*theta[vocabulary.index(t)]
            else:
                pass
        ped[i] = int(pred>0)
#        if ped == d[1]:
#            counter += 1
    return ped



def generateConfusionMatrix(actual, predicted):    
    results = confusion_matrix(actual, predicted)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :',accuracy_score(actual, predicted))
    print('Report : ')
    print(classification_report(actual, predicted) )


vocab = list(train_data.columns)
#vocabulary = vocab[1:]
Test = testClassification(test_data, test, theta, vocab)


generateConfusionMatrix(actual = test_data.iloc[:,1], predicted = Test)
