import nltk	 
import itertools
import couchdb
import operator

from nltk import Text, FreqDist
from nltk import ContextIndex
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder, TrigramCollocationFinder

#read lines of text file
from os import path, listdir
from sys import argv
from uuid import uuid4

import utils as tech
import cPickle
from matplotlib.pyplot import *

path = 'https://acharya:marathon221$@acharya.cloudant.com/'
drug = argv[1]
dbname = 'yt-%s' % drug

server = couchdb.Server(path)
db = server[dbname]

import gensim

print 'Getting magma'
magma = []
for id in db:
	if 'comments' in db[id].keys():
		magma.append(db[id]['comments'])


print 'Got magma' # It's a list of lists where each list is a comment
cPickle.dump(magma,open('magma.pkl','wb'))
print'Pickled'


'''
summary = tech.summarize(magma, processed=True)
print ''.join(summary[:1000])
'''

'''
print nltk.pos_tag(nltk.word_tokenize(' '.join(magma)))

x = nltk.bigrams(magma)
bi_list_one = [one for one,two in x if two == 'cocaine']
bi_list_two = [two for one,two in x if one =='cocaine']

bi_list = list(set(bi_list_one + bi_list_two))
bigram_fdist = FreqDist([word for word in magma if word in bi_list])
tech.plot_word_frequency(bigram_fdist)
savefig('cocaine-bigrams-with-stopwords.png')

clean_bis = tech.remove_stopwords([word for word in magma if word in bi_list])
clean_bis_fdist = FreqDist(clean_bis)
tech.plot_word_frequency(clean_bis_fdist)
savefig('cocaine-bigrams-no-stopwords.png')

fdist = FreqDist(magma)
tech.plot_word_frequency(fdist)
savefig('cocaine-no-stopwords.png')

fdist_clean = FreqDist(tech.remove_stopwords(magma))
tech.plot_word_frequency(fdist_clean)
savefig('cocaine-with-stopwords.png')

doc_id = uuid4().hex
db[doc_id] = {'overall-frequencies':sorted(fdist.items(),key=lambda it: it[1],reverse=True)}
print doc_id

count = len(db)
for i,id in enumerate(db):
	print i/float(count)
	video = db[id]
	if type(video['comments']) is type([]):
		continue

	txt = video['comments'].split()
	
	corpus = Text(txt) 
		
	#First some basic statistics	
	analysis  = {}
	analysis['summary'] = {}
	
	analysis['summary']['len'] = len(txt)
	analysis['summary']['ld'] = tech.lexdiv(video['comments']) #Lexdiv takes a string not a list
	#analysis['summary']['FK'] = tech.FK(video['comments'])
	
	analysis['common-words'] = sorted(FreqDist(corpus).items(),key= lambda it: it[1], reverse=True)
	#analysis['concordant_words'] = corpus.concordance(drug) #Assuming argv1 is the drug name
	analysis['similar-words'] = ContextIndex(txt).similar_words(drug)
	
	#Finds the most frequent words to appear with drug
	bi = BigramAssocMeasures()
	bi_finder = BigramCollocationFinder.from_words(corpus, window_size = 20)	
	analysis['bigrams-general'] = bi_finder.nbest(bi.pmi,30)

	bi_finder.apply_ngram_filter(lambda w1,w2: w1 != drug and w2 != drug)
	analysis['bigrams-drug'] = bi_finder.nbest(bi.pmi,30)
	
	#Finds the two most frequent words to appear with drug
	tri = TrigramAssocMeasures()
	tri_finder = TrigramCollocationFinder.from_words(corpus)
	analysis['trigrams-general'] = tri_finder.nbest(tri.pmi, 30)
	tri_finder.apply_ngram_filter(lambda w1,w2,w3: w1 != drug and w2 != drug and w3 != drug)
	analysis['trigrams_drug']=tri_finder.nbest(tri.pmi, 30)

	video['analysis'] = analysis
	
	
	db[id] = video
'''
