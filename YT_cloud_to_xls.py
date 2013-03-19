import nltk	 
import itertools
import couchdb
import operator
import csv

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

print 'Getting comments'
magma = []
for id in db:
	if 'comments' in db[id].keys():
		#edit the line below to adjust the sets of comments that are 
		comments = db[id]['comments-as-list'][400:600]
		words = [w.encode('utf-8') for w in comments]
		magma.append(words)
		
print magma		
print 'Writing XLS'
resultFile = open("robotrip_400to600_comments.csv",'w')
wr = csv.writer(resultFile, dialect='excel')
for comment_list in magma: 
	for comment in comment_list: 
		wr.writerow([comment])


#print 'Got magma' # It's a list of lists where each list is a comment
#cPickle.dump(magma,open('magma.pkl','wb'))
#print'Pickled'







"""
db = 'https://acharya.cloudant.com/yt-marijuana/'

import gensim

print 'Getting magma'
magma = []
for id in db:
	print id 
	#if 'comments' in db[id].keys():
		#magma.append(db[id]['comments'])


print 'Got magma' # It's a list of lists where each list is a comment
cPickle.dump(magma,open('magma.pkl','wb'))
print'Pickled'
"""
