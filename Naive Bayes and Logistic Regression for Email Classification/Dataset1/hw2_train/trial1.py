import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Function to read files (emails) from the local directory
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            with open(path) as file:
                f = file.read()
#            f = io.open(path, 'r', encoding='latin-1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
#            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

#An empty dataframe with 'message' and 'class' headers
data = DataFrame({'message': [], 'class': []})

#Including the email details with the spam/ham classification in the dataframe
data = data.append(dataFrameFromDirectory('F:/UTD/Machine Learning/Dataset1/hw2_train/train/ham', 'ham'))
data = data.append(dataFrameFromDirectory('F:/UTD/Machine Learning/Dataset1/hw2_train/train/spam', 'spam'))

#Head and the Tail of 'data'
data.head()
print(data.tail())




vectoriser = CountVectorizer()
count = vectoriser.fit_transform(data['message'].values)
print(count)