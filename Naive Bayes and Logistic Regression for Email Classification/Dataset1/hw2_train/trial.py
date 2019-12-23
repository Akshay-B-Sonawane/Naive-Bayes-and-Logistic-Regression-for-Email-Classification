import numpy as np
import re
import pandas as pd
from nltk.tokenize import word_tokenize



with open("0004.1999-12-14.farmer.ham.txt") as f:
    file = f.read()

def extract_words(sentence):
    ignore_words = ['a']
    wordsen = re.sub(r'\d+', "", sentence)
    words = re.sub("[^\w]", " ",  wordsen).split() #nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned  
wor = extract_words(file)

def tokenize_sentences(sentences):
    words = []
    words = word_tokenize(sentences)
    return words


tok = tokenize_sentences(file)


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
                
    return np.array(bag)
bag = bagofwords(file, tok)