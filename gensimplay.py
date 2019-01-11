from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
#from nltk.tokenize import RegexpTokenizer

import os


# Create a corpus from a list of texts
FolderToParse = "BagsOfWords/"
DocList = []
for document in os.listdir( FolderToParse ):
    # load documents
    FileToLoad = FolderToParse + document
    f = open(FileToLoad,'rb')
    words = f.read().decode('ascii', 'ignore')
    f.close()
    words = words.split()
    DocList.append(words)
common_texts = DocList[:5]

#print(len(DocList[3]))
#print(common_texts)
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
#print(common_corpus)
# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=5)

### Now produce probabilities based on the Corpus
LDAvectors = []
for i in range(len(DocList)):
    # first we translate using the dictionary that we have already
    temp = [ common_dictionary.doc2bow(text) for text in DocList[i] ]
    vector = lda[temp[0]]
    LDAvectors.append( vector )

"""
from gensim.test.utils import datapath

# Save model to disk.
temp_file = datapath("model")
lda.save(temp_file)
# Load a potentially pretrained model from disk.
lda = LdaModel.load(temp_file)

# Create a new corpus, made of previously unseen documents.
other_texts = [ ['small', 'brown', 'fast','warm','desert'] ] #,
#    ['human', 'system', 'computer'] ]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]

unseen_doc = other_corpus[0]
vector = lda[unseen_doc] # get topic probability distribution for a document
print(vector)

lda.update(other_corpus)
vector = lda[unseen_doc]
print(vector)
"""
#print(DocList)
