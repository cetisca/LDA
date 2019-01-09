import zipfile
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

FolderToParse = "Data_Part/"
for document in os.listdir( FolderToParse ):
    # load document
    FileToLoad = FolderToParse + document
    # Load text file of wikipedia entry on bird species
    f = open(FileToLoad,'r')
    data = f.read()
    f.close()

    # Separate words into entries in list, while removing punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)
    # make all words lowercase to enable matching for filtering
    words = [token.lower() for token in tokens]
    # remove any stopwords (common words in english language not useful to parse to LDA)
    words = list(filter(lambda token: token not in stopwords.words('english'), words))
    # remove numbers
    words = [token for token in words if not token.isdigit()]
    # remove single letter or two letter words
    words = [token for token in words if len(token)>2]
    #print(words)
    # Lemmatization (collapse different forms of the same word into one)
    words = [WordNetLemmatizer().lemmatize(token) for token in words]

    # save new files
    tosave = document[:-5] + '_BOW.txt'
    f = open(BagsOfWords/tosave,'w')
    f.write(words)
    f.close()
