import csv
from random import shuffle
import utils as tech

import numpy as np
from matplotlib.pyplot import *
import utils as tech
import nltk

import nltk.metrics

import collections

filename = '/Users/andymckenzie/dropbox/ToxTweet/APIs/YouTube/marijuana_20_comments.csv'

with open(filename,'rU') as f:
	reader = csv.reader(f,delimiter=',')
	data = [row for row in reader]

def get_features(comment,medi,mad,medi_ld,mad_ld):
	return {	'long': 'yes' if len(comment.split()) > (4*mad+medi) else 'no',
				'explicit' : 'yes' if any([keyword in comment.split() for keyword in ['marijuana','weed']]) else 'no',
#				'FX': 'yes' if any([keyword in comment.split() for keyword in ['brain','cancer','creative','counseling','migraines']]) else 'no',
				'rants': 'yes' if any([keyword in comment.split() for keyword in ['govt','government']]) else 'no'}
#format data
print len(data)
data = data[:960]	
#shuffle data
shuffle(data)

lens = [len(tech.clean(comment[2]).split()) for comment in data]
medi = np.median(lens)
mad = np.median(abs(medi-lens))

lds = [tech.lexdiv(comment[2]) for comment in data]
medi_ld = np.median(lds)
mad_ld = np.median(abs(medi_ld-lds))
print medi_ld,mad_ld

#Lex div seems to do nothing more than length does

feature_sets = [(get_features(tech.clean(comment[2]),medi,mad,medi_ld,mad_ld),comment[1]) for comment in data]
train_set,test_set = feature_sets[:len(feature_sets)/2],feature_sets[len(feature_sets)/2:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.show_most_informative_features(15)

#print nltk.classify.accuracy(classifier,test_set)

#error analysis
errors = []
for _,tag,comment in data[len(data)/2:]:
	guess = classifier.classify(get_features(tech.clean(comment),medi,mad,medi_ld,mad_ld))
	if guess != tag:
		errors.append((tag,guess,comment))

def rate(classifier,filename,medi,mad,medi_ld,mad_ld):
	#For now, just assuming the text is csv
	results = classifier
	with open(filename,'rU') as f:
		reader = csv.reader(f,delimiter=',')
		data = [(get_features(tech.clean(row[2]),medi,mad,medi_ld,mad_ld),row[0]) for row in reader]
	print nltk.classify.accuracy(classifier,data)

testing_name = '/Volumes/My Book/Dropbox/ToxTweet/APIs/YouTube/NGEP.csv'
with open(testing_name,'rU') as f:
	reader = csv.reader(f,delimiter=',')
	testfeats = [(get_features(tech.clean(row[2]),medi,mad,medi_ld,mad_ld),row[0]) for row in reader]

#-precision and recall

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

negfeats = [datum for datum in feature_sets if datum[1] == -1]
posfeats = [datum for datum in feature_sets if datum[1] == 1]


#rate(classifier,testing_name,medi,mad,medi_ld,mad_ld)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
    
print 'pos precision:', nltk.metrics.precision(refsets['1'], testsets['-1'])
print 'pos recall:', nltk.metrics.recall(refsets['1'], testsets['1'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['1'], testsets['1'])
print 'neg precision:', nltk.metrics.precision(refsets['-1'], testsets['-1'])
print 'neg recall:', nltk.metrics.recall(refsets['-1'], testsets['-1'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['-1'], testsets['-1'])

torate = '/Volumes/My Book/Dropbox/ToxTweet/APIs/YouTube/nonrated_marijuana_YT_comments.csv'
outname = '/Volumes/My Book/Dropbox/ToxTweet/APIs/YouTube/nonrated_marijuana_YT_comments_CLASSIFIED_NO_CANCER.csv' 

with open(torate,'rU') as f:
	reader = csv.reader(f,delimiter=',')
	dat = [get_features(tech.clean(row[0]),medi,mad,medi_ld,mad_ld) for row in reader]
	classified = [classifier.classify(feat) for feat in dat]

with open(outname,'wb') as f:
	writer = csv.writer(f)
	writer.writerows(zip(classified,dat))
