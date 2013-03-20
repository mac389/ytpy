import cPickle
from nltk import WordNetLemmatizer as wnl
from nltk import FreqDist
import numpy as np
from scipy.io import savemat
'''
stopwords = [word.rstrip('\n') for word in open('stopwords','rb').readlines()]
comments = cPickle.load(open('everything.pkl','rb'))


num2word = {'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine',
			'fuckn':'fucking'}

list_comments = [comment.split() for comment in comments]
print 'Converting to ASCII'
list_comments = [[word.encode('ascii','ignore') for word in comment] for comment in list_comments]
print 'Converting all single numbers to string'
list_comments = [[num2word[word.strip()] if word.strip() in num2word else word for word in comment] for comment in list_comments]
print 'Ignoring strings of numbers'
list_comments = [[word for word in comment if word.isalpha()] for comment in list_comments]
print 'Ignoring stopwords'
list_comments = [[word for word in comment if word not in stopwords] for comment in list_comments]

print 'Lemmatizing'
list_comments = [[wnl().lemmatize(word) for word in comment] for comment in list_comments]

cPickle.dump(list_comments,open('everything.processed','wb'))
'''

#Forget words that occur only 1% of the time

comments = cPickle.load(open('everything.pkl','rb'))
'''
all_words = sum(list_comments,[])
D = len(list_comments)
print 'There are %d documents' % D
print 'Getting Frequencies'
freqs = [FreqDist(comment) for comment in list_comments]

#Normalize to account for comments of different length
for fdist in freqs:
	N = fdist.values()[0]
	for word,frequency in fdist.iteritems():
		fdist[word] = frequency/float(N)
print freqs

stopwords = [word.rstrip('\n') for word in open('stopwords','rb').readlines()]
print 'Got stopwords'
print 'Creating vectorizer'
vectorizer = CountVectorizer(strip_accents='ascii',stop_words = stopwords)
print 'Transforming'
vectorizer.fit_transform(comments)
cPickle.dump(vectorizer,open('tfidfvectorizer.pkl','wb'))
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorizer = cPickle.load(open('tfidfvectorizer.pkl','rb'))
print 'Depickled Count Vectorizer'
smatrix = vectorizer.fit_transform(comments)
smatrix.todense().tofile('dense-tfidf.mat')
np.savetxt('sparsematrix-shape.data',smatrix.shape,delimiter='\t')
print 'Saved a dense copy of the matrix'

tfidf = TfidfTransformer(norm='l2')
tf_idf_matrix = tfidf.fit_transform(smatrix)
savemat('tfidf_matrix.mat',mdict={'tfidf':tf_idf_matrix.todense()})