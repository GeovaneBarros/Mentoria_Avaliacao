import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import words, stopwords
import numpy as np

nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def y_encoder(y, dict_change):
    new_x = []
    for item in y:
        for key in dict_change:
            if key == item:
                new_x.append(dict_change[key])
    return new_x


def change_words(X, dict_change):
    new_x = []
    for item in X:
        item = item.lower()
        for key in dict_change:
            if key in item:
                item.replace(key, dict_change[key])
        new_x.append(item)
    return new_x

def basic_preprocessing(X):
    new_x = []
    for item in X:
        item = item.lower()    
        item = re.sub("\W", ' ', item)
        item = re.sub(" +", ' ', item)
        word_tokens = nltk.word_tokenize(item)
        tokens = [x for x in word_tokens if x not in stopwords.words('english')]  
        new_x.append(' '.join(tokens))
    return new_x
    
def lemmatizer(X):
    lemmatizer = WordNetLemmatizer()
    new_x = []
    for i in X:
        tokens = nltk.word_tokenize(i)
        for j in range(len(tokens)):
            tokens[j] = lemmatizer.lemmatize(tokens[j])
        new_x.append(' '.join(tokens))
    return new_x
