import couchdb
import utils as tech
from random import random
import cPickle
from FK import *
import json
from nltk import FreqDist
import matplotlib.pyplot as plt
from numpy import *

from matplotlib import rcParams
rcParams['axes.linewidth'] = 2
rcParams['mathtext.default']='bf'
rcParams['xtick.labelsize']='large'
rcParams['ytick.labelsize']='large'
#Compare FK and LZ for heroin, cocaine, and marijuana, coffee, and beer. 

sampling_fraction = 0.1

dblist =  ['marijuana','coffee'] #['heroin','cocaine']have to do heroin separately because the database is so small
'''
server = couchdb.Server()
for id in server:
	print id
	if id in dblist:
		db = server[id]
		sample = []
		print db
		for key in tech.couchdb_pager(db):
			if random() < sampling_fraction:
				try:
					sample.append(tech.clean(db[key]['text']))
				except KeyError:
					try:
						text = ' '.join([res['text'] for res in db[key]['results']])
						sample.append(tech.clean(text))
					except:
						continue
		cPickle.dump(sample,open(id+'-sample.pkl','wb'))
		
#-
'''
processing_list = ['marijuana','coffee','heroin','cocaine']
analysis = {}
for drug in processing_list:
	'''
	data =  ' '.join(cPickle.load(open('%s-sample.pkl'%drug,'rb'))).split()
	#Pickle returns a list, all the tech methods prefer strings
	lens = map(len,data)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(lens,bins=50,range=[0,40])
	plt.show()
	
	#get LZ
	lzc = tech.LZC(data)
	FK = flesch_kincaid_level(data)
	print drug
	analysis[drug] = {'LZC':lzc,'FK':FK}
	
	print analysis[drug]
	
	fdist = FreqDist(tech.remove_stopwords(data))
	cPickle.dump(fdist,open('%s-fdist.pkl'%drug,'wb'))
	
	
	data =  ' '.join(cPickle.load(open('%s-sample.pkl'%drug,'rb'))).split()
	#Pickle returns a list, all the tech methods prefer strings
	lens = map(len,data)
	'''
	fdist = cPickle.load(open('%s-fdist.pkl'%drug,'rb'))
	'''
	wordfreq= [(len(word),freq) for word,freq in fdist.iteritems()]
	lens = arange(3,20)
	final_data =[]
	for length in lens:
		final_data.append(sum(wf[1] for wf in wordfreq if wf[0] == length))
	final_data = array(final_data).astype(float)
	final_data /= float(final_data.sum())
	final_data *= 100
	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111)
	ax.semilogy(lens,final_data,'k',linewidth=3,label=r'\textbf{%s tweets}'%drug.capitalize())
	plt.hold(True)
	tech.adjust_spines(ax,['bottom','left'])
	ax.semilogy(lens,21.2*.73**(lens-3),'k--',linewidth=3, label=r"\textbf{Zipf's law}")
	ax.set_xlabel(r'\textbf{Word Length}', fontsize=20)
	ax.set_ylabel(r'\textbf{Frequency}', fontsize=20)
	ax.legend(loc='lower left', frameon=False)
	
	plt.show()
	'''
	tech.plot_word_frequency(fdist,cutoff=20,drug=drug,poster=True)
	plt.show()
	
#json.dump(analysis,open('acmt-analysis.json','wb'))
					