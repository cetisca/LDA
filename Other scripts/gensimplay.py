#from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
#from nltk.tokenize import RegexpTokenizer
import os

# Create a list with all texts
FolderToParse = "BagsOfWords/"
DocList = []
for document in os.listdir( FolderToParse ):
    # load documents
    FileToLoad = FolderToParse + document
    f = open(FileToLoad,'rb')
    words = f.read().decode('ascii', 'ignore')
    f.close()
    words = words.split() # tokenize
    DocList.append(words)
    
DocList = DocList[:100]
# Load the names of species
with open('list_of_species.txt', encoding='utf-8', errors='ignore') as f:
    Names = f.readlines()
Names = [x.strip() for x in Names]

# Create dictionary [ID to word]
common_dictionary = Dictionary(DocList)
# Create text to words mappings & count
common_corpus = [common_dictionary.doc2bow(text) for text in DocList]
print("We have loaded {0} documents and {1} words in total!".format(len(DocList), len(common_dictionary)))

# Train the model on the corpus.
N_Topics = 10
lda = LdaModel(common_corpus, num_topics=N_Topics,alpha='asymmetric', random_state=1)
# Now produce probabilities based on the Corpus
LDAvectors = []
for i in range(len(DocList)):
    # first we translate using the dictionary that we have already
    temp = [ common_dictionary.doc2bow(text.split()) for text in DocList[i] ]
    vector = lda[temp[0]]
    LDAvectors.append( vector )
print('LDA is complete!')

##############
# Clustering #

####################
# 1. LDA by gensim # 
# the number of clusters is N_Topics
ClustersLDA = { i:[] for i in range(N_Topics)}# initialize the dictictionary of clusters
ClustersNames = { i:[] for i in range(N_Topics)} 
Labels = []
for i in range(len(DocList)):
    distr = [ x[1] for x in LDAvectors[i]]
    # find the argmax{distr} - ATTENTION: ties ???
    label = distr.index(max(distr))
    ClustersLDA[label].append(i)
    ClustersNames[label].append(Names[i])
    Labels.append( label )
    
####################
# 3. LDA by Scikit #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

# we need the raw texts as input for this method
FolderToParse = "Data_Part/pagecontent_as_text/"
RawTexts = []; N=0
for document in os.listdir( FolderToParse ):
    # load document
    FileToLoad = FolderToParse + document
    # Load text file of wikipedia entry on bird species
    f = open(FileToLoad,'rb')
    # we ignore non-printable (strange) letters and symbols !
    text = f.read().decode('ascii', 'ignore')
    f.close()
    RawTexts.append(text)
    
LDAsklearn = LatentDirichletAllocation(n_topics = N_Topics, max_iter=50, random_state=1)
vectorizer = TfidfVectorizer(max_df=0.3, max_features=60000, min_df=1, stop_words='english', use_idf=True)
print("n_samples: %d, n_features: %d" % X.shape)
X = vectorizer.fit_transform(RawTexts)
LDAsklearn.fit(X)
LDA_SKLvectors = LDAsklearn.transform(X)
# Clustering - Black and white approach, as before
ClustersSKL = { i:[] for i in range(N_Topics)}
LabelsSKL = []
for i in range(len(DocList)):
    distr = list(LDA_SKLvectors[i])
    # find the argmax{distr} - ATTENTION: ties ???
    label = distr.index(max(distr))
    ClustersSKL[label].append(i)
    LabelsSKL.append( label )
    
#############
# 2. KMeans #

# Dimensionality reduction
svd = TruncatedSVD(1000)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
km = KMeans(n_clusters=N_Topics, init='k-means++', max_iter=100, n_init=1, random_state=1)
print("Clustering sparse data with %s" % km)
km.fit(X)

# create the clusters
ClustersKM = { i:[] for i in range(N_Topics)}
LabelsKM = list(km.labels_)
for i in range(len(DocList)):
    ClustersKM[LabelsKM[i]].append(i)
    
# in order to compare, we first need to associate the different clusters!
"""
print("Homogeneity: %0.3f" % metrics.homogeneity_score(Labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(Labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(Labels, km.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(Labels, km.labels_))
print("Mutual information: %.3f" %metrics.mutual_info_score(ClustersKM.values(),Clusters.values()))

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