# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 01:11:11 2019

@author: giork
"""
###################
# Initiliazations #
import numpy as np
import os
from scipy.sparse import load_npz, save_npz
from scipy.sparse import dia_matrix, csr_matrix, csc_matrix
from time import time
from scipy.spatial.distance import pdist, squareform
##################
# Load Documents #
FolderToParse = "BoW_100random/"# "BagsOfWords/" # 
DocList = []
totalwords = 0 # vaariable for sanity check
for document in os.listdir( FolderToParse ):
    # load documents
    FileToLoad = FolderToParse + document
    f = open(FileToLoad,'rb')
    words = f.read().decode('ascii', 'ignore')
    f.close()
    words = words.split()
    DocList.append(words)
    # counter for sanity checks
    totalwords += len(words)

Vocabulary = np.unique(np.concatenate(DocList))
N_D = len(DocList) # number of documents
N_V = len(Vocabulary)  # number of vocabulary - all available words

print("We have {0} documents and {1} words in total!".format(N_D, N_V))
print("Total number of words found= %d" % totalwords )
###############################
# Create Term-Document matrix #

#TDM = csr_matrix((N_V,N_D))
TDM = np.zeros((N_V,N_D)) # we start by a full matrix and then transform it
t0 = time()
# create doc-word list of lists
for doc in range(N_D):
    temp = np.unique( DocList[doc] ) # get the different words on this document
    for i in range( len(temp) ):
        word = temp[i] 
        count = len([ x for x in DocList[doc] if x == word])
        # we must get the index of this word in the (total) corpus
        TDM[ np.where(Vocabulary == word) , doc] = count
    # progress check
    if ((doc+1) % 500) == 0 :
        print('More than {0} documents have been processed! Rate = {1}'.format(doc+1, (time()-t0)/doc))

# sanity check
for doc in range(N_D):
    if sum(TDM[:,doc])!=len(DocList[doc]):
        print("Doc-{0} has a problem!".format(doc))
        
TDM = csr_matrix(TDM)
print("The Term-Doc matrix is {0:.2f}% dense.".format(csr_matrix.count_nonzero(TDM)/np.prod(TDM.shape)*100))
# Save ?
filetosave = 'Term_Doc_Matrix_'+str(N_D)+'.npz'
save_npz(filetosave,TDM)

########################
# Create TF-IDF Matrix #
print("Proceeding to the creation of TF-IDF matrix...")
# create tf-idf matrix
# first compute the frequencies of each word among documents - sum of rows in TDM :
IDF = TDM.sum(axis=1)
IDF = np.array( [np.log(N_D/tf) for tf in IDF] ) # trick to get rid of the (1,N_V) dimension thing
IDF = [x[0][0] for x in IDF]
# create sparse diag matrix :
spdiag = csr_matrix( (IDF, range(N_V),range(N_V+1)) )
TF_IDF = TDM.transpose()*spdiag
# Save ?
filetosave = 'TFIDF_'+str(N_D)+'.npy'
np.save(filetosave,TF_IDF)

#####################
# Compute Distances #
print("Now calculating distances...")
Distances = pdist(TF_IDF.todense(), 'euclidean')
Distances = squareform(Distances) # this is a full matrix (and heavy!)
# Save ?
filetosave = 'DocDistances_'+str(N_D)+'.npy'
np.save(filetosave, Distances)

"""
# an alternative (and slower?) way
from scipy.sparse.linalg import norm
Distances = np.empty((N_D,N_D))
Distances = []
t0=time()
for i in range(2):
#    Distances[i][i+1:] = [ norm(TF_IDF[:,i] - TF_IDF[:,j]) for j in range(i+1,N_D)]
    Distances.append( [ norm(TF_IDF[:,i] - TF_IDF[:,j]) for j in range(i+1,N_D)] )
print((time()-t0)/10)
"""

