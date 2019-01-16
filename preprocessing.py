#!/usr/bin/env python3
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

FolderToParse = "Data_Part/"
FolderBOW = "BagsOfWords/"
# Create a corpus for all documents
Corpus = []; N=0
for document in os.listdir( FolderToParse ):
    # load document
    FileToLoad = FolderToParse + document
    # Load text file of wikipedia entry on bird species
    f = open(FileToLoad,'rb')
    # we ignore non-printable (strange) letters and symbols !
    text = f.read().decode('ascii', 'ignore')
    f.close()

    # Separate words into entries in list, while removing punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
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

    words = set(words) # get rid of duplicates
    # save new files
    tosave = document[:-5] + '_BOW.txt'
    f = open(FolderBOW + tosave,'w')
    for w in words:
        #ww = str( w+' '.encode("ascii", "replace") )
        f.write(w + ' ')#.encode("ascii")
        Corpus.append(w)
    f.close()
    N += 1

print("Basic preprocessing is done! {0} documents were found.".format(N))

Corpus = set(Corpus)
f = open("Corpus.txt", 'w')
for w in Corpus:
    f.write(w+' ')
f.close()
V = len(Corpus)
print("The vocabulary has size = ", V)

#####
# Create a term-document matrix
import numpy as np
import scipy
TDM = scipy.sparse.csr_matrix((V,N))
