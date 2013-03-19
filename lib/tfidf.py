import csv
import cPickle

import numpy as np
import utils as tech
import Graphics as artist
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from scipy.sparse.linalg import svds

from sklearn import metrics
from sklearn.cluster import KMeans
from numpy.linalg import eig
from matplotlib import cm
from nltk import FreqDist
from numpy.random import random_integers
from gensim import corpora, models, similarities

from nltk.tokenize import word_tokenize
from nltk import Text
#from sortUtils import cluster
import utils as tech
data = []
filename = '/Volumes/My Book/Dropbox/ToxTweet/APIs/YouTube/nonrated_marijuana_YT_comments_CLASSIFIED_NO_CANCER.csv'
with open(filename,'rU') as f:
	reader = csv.reader(f,delimiter=',')
	for row in reader:
		try:
			data.append((tech.clean(row[4]),row[0]))
		except:
			pass

'''
neg = [datum[0] for datum in data if datum[1] == '-1']
pos = [datum[0] for datum in data if datum[1] == '1']
'''

all_together = [word_tokenize(datum[0]) for datum in data if datum[1] == '1']
as_one = []
for datum in data:
	as_one.extend(word_tokenize(datum[0]))
print as_one
#LSA

'''Standard pre-processing
	1. Remove stopwords
	2. Remove words that occur less than threshold times
	'''
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

threshold = 5
count = FreqDist(Text(as_one))
texts = [[word for word in text if count[word] > threshold] for text in all_together]
print 'Finish preprocessing for LSA'
dictionary = corpora.Dictionary(texts)
dictionary.save('/Volumes/My Book/yt-mj-pos-lsa.dict')

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/Volumes/My Book/yt-mj-pos.mm', corpus)
tfidf = models.TfidfModel(corpus)


lsi=models.LsiModel(corpus=tfidf[corpus],id2word=dictionary)
lsi.print_topics(num_topics=20),'$$$$'


lda =models.ldamodel.LdaModel(corpus=corpus,id2word = dictionary, num_topics = 1000, passes=10) 	


print 'made LDA'		
lda.save('mj-pos.lda')
lda.show_topics(50)

'''
vectorizer = CountVectorizer()
vectorizer.fit_transform(all_together)
freq_term_matrix = vectorizer.transform(all_together)

to_tfidf = TfidfTransformer()
to_tfidf.fit(freq_term_matrix)

tfidf = to_tfidf.transform(freq_term_matrix)
k=90
u,s,vt = svds(tfidf,k=k)
S = np.diag(s)

artist.eig_spectrum(s[::-1]**2, show=True,save=False)
artist.scree_plot(s[::-1]**2, show=True,save=False)
score = np.dot(u.T,tfidf.todense())

word_score = np.dot(u.T,tfidf.todense())
doc_score = np.dot(tfidf.todense(),vt.T)

localpath = '/Volumes/My Book/Dropbox/ToxTweet/APIs/YouTube/marijuana-yt'
(models,silhouettes) = cluster(doc_score, preprocess=False)
CLUSTER = localpath+'-cluster.pkl'
res = {'models':models,'silhouettes':silhouettes}
cPickle.dump(res,open(CLUSTER,'wb'))
'''