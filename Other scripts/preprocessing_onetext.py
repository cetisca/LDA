#!/usr/bin/env python3
import zipfile
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

'''
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
'''

# Load text file of wikipedia entry on bird species
f = open('Corpus.txt','r')
corpus = f.read()
f.close
# Separate words into entries in list, while removing punctuation
tokenizercrp= RegexpTokenizer(r'\w+')
corpustokens = tokenizercrp.tokenize(corpus)

# Load corpus

f = open(' 6 .txt','r')
data = f.read()
f.close
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

# Lemmatization (collapse different forms of the same word into one)
words = [WordNetLemmatizer().lemmatize(token) for token in words]

data_pp = [corpustokens]

id2word = corpora.Dictionary(data_pp)

texts = [words]

corpus = [id2word.doc2bow(text) for text in texts]

# print(corpus[:1])
# print(id2word[0])
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=2,
                                           random_state=75,
                                           update_every=1,


                                           alpha='auto',
                                           per_word_topics=True)


print(lda_model.print_topics())
doc_lda = lda_model[corpus]

coherence_model_lda = CoherenceModel(model=lda_model, texts=[words], dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print(coherence_lda)
