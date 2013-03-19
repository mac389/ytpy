import nltk	 
import itertools
from nltk import Text, FreqDist
from nltk import ContextIndex
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder, TrigramCollocationFinder


#read lines of text file
from os import path
from os import listdir
import utils as tech
from nltk import bigrams
import csv
drug = 'robotrip'
import cPickle

fd = nltk.FreqDist()

#wordlist = open('list_body_parts.list.txt','rb').read().splitlines()
data = []
filename = '/Users/andymckenzie/Dropbox/ToxTweet/APIs/YouTube/nonrated_robotrip_comments.csv'
with open(filename,'rU') as f:
	reader = csv.reader(f,delimiter=',')
	for row in reader:
		try:
			data.append((tech.clean(row[4]),row[0]))
		except:
			pass
	
neg = [datum[0] for datum in data if datum[1] == '-1']
pos = [datum[0] for datum in data if datum[1] == '1']

neg_flat = tech.remove_stopwords((' '.join(neg)).split())
pos_flat = tech.remove_stopwords((' '.join(pos)).split())
cPickle.dump((pos_flat,neg_flat),open('%s-flat-yt.pkl'%drug,'wb'))

print neg_flat[0]

'''
txt = []	
with open('/Volumes/My Book/Dropbox/toxtweet/APIs/YouTube/nonrated_marijuana_YT_comments_CLASSIFIED_NO_CANCER.csv','r') as f:
	text = f.read().splitlines()
	text = [comment.split() for comment in text]
	text = [item for sublist in text for item in sublist]
	txt.extend(text)
'''		


fdist = {'pos': FreqDist(pos_flat),'neg':FreqDist(neg_flat)}
fdist['pos'].keys()[:10]
fdist['neg'].keys()[:10]
bigrams = {'pos':tech.bigrams(pos_flat,drug),'neg':tech.bigrams(neg_flat,drug)}
cPickle.dump((fdist,bigrams),open('%s-fdist-yt.pkl'%drug,'wb'))
#Text(pos_flat).bigrams()
#Text(neg_flat).bigrams()

'''

#print 'Words similar to ' + drug + ' using Text method'
#Finds other words which appear in the same contexts


#print 'Words similar to ' + drug + ' using ContextIndex method'
#ytcomments2 = ContextIndex(txt)

print '####'
#sim = ytcomments2.similar_words(drug)

print '---'
corpus.collocations()
print '----'
#Finds the most frequent words to appear with drug
bi = BigramAssocMeasures()
bi_finder = BigramCollocationFinder.from_words(corpus, window_size = 20)
top_general = bi_finder.nbest(bi.pmi,30)
bi_finder.apply_ngram_filter(lambda w1,w2: w1 != drug and w2 != drug)
print top_general
#top_bi = bi_finder.nbest(bi.pmi, 30)

#Finds the two most frequent words to appear with drug
tri = TrigramAssocMeasures()
tri_finder = TrigramCollocationFinder.from_words(corpus)
tri_finder.apply_ngram_filter(lambda w1,w2,w3: w1 != drug and w2 != drug and w3 != drug)
#top_tri = tri_finder.nbest(tri.pmi, 30)


for file in files:
	with open(file,'rb') as f:
		text = f.readlines()
		lists = ' '.join(text)
		document = nltk.tokenize.word_tokenize(lists)
		fdist = nltk.FreqDist(Text(document))
		print fdist
'''
#for word in wordlist: 
#	print word,'\t', fdist.freq(word)
	
	
	
