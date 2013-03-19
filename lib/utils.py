from numpy import *
from scipy import *
from matplotlib.pyplot import *

from readabilitytests import ReadabilityTool
from zlib import compress
from numpy.random import rand

from itertools import product

import couchdb
import cPickle	
import xlwt
import csv
from nltk import Text, FreqDist, ContextIndex, tokenize, pos_tag, word_tokenize

from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder, TrigramCollocationFinder
from termcolor import colored,cprint

from gensim import corpora, models, similarities

from scipy.stats import scoreatpercentile

from collections import defaultdict

from matplotlib import rcParams
rcParams['text.usetex'] = True

#------Adaptation of Flesch-Kincaid grade level for tweets-----------
#Instead of words per sentence, we consider words per tweet
#--------------------------------------------------------------------

delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
verboten = ['@','RT','http://','#']

def summarize(text, num_sentences=10, processed=False):
	''' Luhn's 1958 algorithm- return only the sentences that have the most important words in the order
	   they occurred in the text
	'''
	
	words = remove_stopwords(flatten(text)) if processed else remove_stopwords(flatten([clean(sentence).split() 
													 for sentence in tokenize.sent_tokenize(text)]))
	
	
	# Pooling comments from all videos	
	important_words = FreqDist(words).keys()[:50]	
	#add sentences with most frequent words
	abstract = [sentence for sentence in text for important_word in important_words if important_word in sentence][:num_sentences]
	paragraph = ' '.join(flatten(text))
	abstract.sort(lambda une,deux: paragraph.find(''.join(une)) - paragraph.find(''.join(deux))) #<-- Python sorts lists in place.		
	return " ".join(flatten(abstract))

def FK(tweet):
	return ReadabilityTool(clean(tweet)).FleschKincaidGradeLevel()

def should_exclude_video(comments,stopwords):
	#loops are fine 
	description = set(clean(comments).split())
	stopwords = set(clean(stopwords).split())
	
	extra_words = {'music video','honey cocaine','johnny cash','jimi hendrix','official video'}
	
	description_by_two = set([''.join(two) for two in product(list(description)[::2],list(description)[1::2])])
	
	if description & stopwords or description_by_two & extra_words:
		first = ' '.join(description & stopwords)
		second = ' '.join(description_by_two & extra_words)
		cprint(' '.join([first,second]),'magenta')
		return True
	else:
		return False

def bigrams(comments,drug=None,cutoff=95):
	bi = BigramAssocMeasures()
	finder = BigramCollocationFinder.from_words(comments)
	res = {}
	res['general'] = {}
	res['general']['data'] = finder.score_ngrams(bi.likelihood_ratio)
	cutoff_idx = scoreatpercentile([score[1] for score in res['general']['data']],cutoff)
	res['general']['cutoff'] = cutoff_idx
	if drug:
		bi_finder.apply_ngram_filter(lambda w1,w2: drug not in w1 and drug not in w2)
		res['with-%s'%drug]  = {}
		res['with-%s'%drug]['data'] = finder.score_ngrams(bi.likelihood_ratio)
		cutoff_idx = scoreatpercentile([score[1] for score in res['with-%s'%drug]['data']],cutoff)
		res['with-%s'%drug]['cutoff'] = cutoff_idx
	return res

def trigrams(comments,drug):
	tri = TrigramAssocMeasures()
	tri_finder = TrigramCollocationFinder.from_words(Text(comments))
	tri_finder.apply_ngram_filter(lambda w1,w2,w3: drug not in w1 and drug not in w2 and drug not in w3)
	return tri_finder.nbest(tri.pmi,30)

def context(comments,drug):
	return ContextIndex(comments).similar_words(drug)

def LSA(texts, threshold=5, claen=True):
	'''Standard pre-processing
		1. Remove stopwords
		2. Remove words that occur less than threshold times
		'''
	import logging
	logging.basicConfig(format='%(message)s', level=logging.INFO)
	
	if clean:
		texts = map(remove_stopwords,texts)
	
	counts = [FreqDist(text) for text in texts]
	texts = [[word for word in text if count[word] > threshold] for count,text in zip(counts,texts)]
	
	print 'Finish preprocessing for LSA'
	dictionary = corpora.Dictionary(texts)
	dictionary.save('/Volumes/My Book/cocaine-lsa.dict')
	
	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('/Volumes/My Book/cocaine.mm', corpus)
	tfidf = models.TfidfModel(corpus)
	

	lsi=models.LsiModel(corpus=tfidf[corpus],id2word=dictionary)
	lsi.print_topics(num_topics=20),'$$$$'
	
	
	lda =models.ldamodel.LdaModel(corpus=corpus,id2word = dictionary, num_topics = 1000, passes=10) 	
	
	
	print 'made LDA'		
	lda.save('cocaine.lda')
	lda.show_topics(50)
	
	
table = {'assholedont': 'asshole', 'fuckin':'fucking','didnt':'did','riosfamily': 'family',
		 'fk':'fucking','mistakesbut':'mistakes','bluesbut':'blues','fukinmovie':'fucking',
		 'stupidand':'stupid','u':'you','isnt':'not'}


def lemmingtize(utterance, pos=True):
	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import wordnet
	lmtzr = WordNetLemmatizer()
	if utterance is []:
		return ' '.join(flatten([lemmingtize(sentence) for sentence in utterance]))
	elif utterance is '' or isinstance(utterance,unicode):
		print '.'
		utterance = clean(utterance)
		for key,value in table.iteritems():
			text = utterance.replace(key,value)
		text = word_tokenize(utterance)
		tagged = pos_tag(text)
		morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
		tagged = [(word,morphy_tag[part] if part in morphy_tag else wordnet.NOUN) for word,part in tagged]
		return ' '.join(lmtzr.lemmatize(word,part) for word,part in tagged)
	else:
		print 'Type %s not understood' % type(utterance)

def clean(tweet,keepTags = False):
	if not keepTags:
		handles = set(filter(lambda x: 1 in [symbol in x for symbol in verboten], tweet.split(' ')))
		tweet= ' '.join([filter(lambda x: ord(x)<128,word) for word in tweet.split(' ') if word not in handles])
        return ' '.join([word.translate(None, delchars).lower() for word in tweet.encode('ascii','ignore').split()])

def remove_stopwords(utterance, languages=['english','spanish','french']): 
	#languages is a list
	print 'HHHHHHHHHHHHHHHHHHHHHHHHHHH'
	from nltk.corpus import stopwords
	my_stopwords = map(lambda word: word.rstrip('\n'),open('stopwords','rb').readlines())
	allowed_languages = {'english','spanish','french'} #This is a set
	#Serial processing is least obfuscated
	for language in set(languages).intersection(allowed_languages):
		utterance = [word for word in utterance if word not in stopwords.words(language)]
	utterance = [word for word in utterance if word not in my_stopwords]
	
	return utterance

def collocations(data, cutoff=20, intervening_words = 0):
	finder = BigramCollocationFinder.from_words(data, window_size = intervening_words + 2)	
	if intervening_words ==0:
		return finder.nbest(BigramAssocMeasures().pmi, cutoff)	
	else:
		return finder.nbest(BigramAssocMeasures().likelihood_ratio,cutoff)

def plot_word_frequency(freqDist,cutoff=75,normalized=True,drug=None, poster = False):
	if poster:
		rcParams['axes.linewidth']=2
		rcParams['mathtext.default']='bf'
		rcParams['xtick.labelsize']='large'
		rcParams['ytick.labelsize']='large'
	labels,vals = zip(*freqDist.items())
	labels = list(labels)
	vals = array(vals).astype(float)
	vals /= vals.sum() #<--- Normalize values

	fig = figure(figsize=(6,7))
	fig.subplots_adjust(left=0.08,right=0.98, top = .95, bottom=0.18)
	ax = fig.add_subplot(111)
	
	
	line, = ax.plot(vals[:cutoff], 'k', linewidth=2)
	adjust_spines(ax,['bottom','left'])	
	ax.set_ylim(ymax=0.20)

	line.set_clip_on(False)
	
	ax.set_ylabel(r'\Large \textbf{Frequency}')
	
	ax.set_xticks(arange(cutoff))
	ax.set_xticklabels([r'\Large \textbf{%s}'%label for label in labels[:cutoff]], rotation=90)

	ax.annotate(r"\Large \textbf{%s}" %drug.capitalize(), xy=(.5, .5),  xycoords='axes fraction',horizontalalignment='center', verticalalignment='center')
	tight_layout()


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
	
def validate(filename):
	import string
	MAX_LENGTH = 30 
	valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
	return filter(lambda char: char in valid_chars,filename)[:MAX_LENGTH]+'.txt'

def lexdiv(tweet):
	return round(len(set(clean(tweet)))/float(len(clean(tweet))+0.001),3)

def LZC(tweet):
  return float(len(compress(clean(tweet))))/len(clean(tweet)) if len(clean(tweet)) > 0 else -1

def rand_chars(length):
	import random
	import string
	
	return ''.join(random.choice(string.letters) for i in range(length))


def grab_locs_from(filename):
	if filename.endswith('.xls'):
		from xlrd import open_workbook
		wb = open_workbook(filename)
		#ZIP code is col 1, Lat is col 4, long is col 5
		#grab first sheet
		sh = wb.sheet_by_index(0)
		zipcodes = sh.col_values(0)
		latitudes = sh.col_values(3)
		longis = sh.col_values(4)
	else: #Assume it is in CSV format
		data = genfromtxt(filename,delimiter=',', dtype = str)
		zipcodes = data[:,0]
		latitudes = data[:,3]
		longis = data[:,4]
	return {str(int(zipcode)):','.join([str(lat),str(lon)]) for zipcode,lat,lon in zip(zipcodes,latitudes,longis)}


def test():
	tweet = 'The quick brown RT 12. jumped over l3zy f)z'
	print clean(tweet)
	print lexdiv(tweet)
	print FK(tweet)
	print LZC(tweet)

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv

def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
           key = key.encode('utf-8')
        if isinstance(value, unicode):
           value = value.encode('utf-8')
        elif isinstance(value, list):
           value = _decode_list(value)
        elif isinstance(value, dict):
           value = _decode_dict(value)
        rv[key] = value
    return rv

def flatten(lst):
	return [item for sublist in lst for item in sublist]

def broadcast(x, idx,count=False):
	print x if not count else idx
	return x

def score(tweet, topics,weights, percentile = 99): #words is a list of lists. Each inner list is a topic
	from scipy.stats import percentileofscore
	answer = []
	for i,topic in enumerate(topics):
		score = 0
		for j,word in enumerate(topic):
			if word in tweet: #Be careful, this is Python but only works if word and tweet are the same data type
				score += float(weights[i][j])
		answer.append((i,round(score,3)))
	scores = [x[1] for x in answer]
	return sorted([datum for datum in answer if percentileofscore(scores,datum[1]) >= percentile] , key = lambda x:x[1], reverse=True)

#
#From: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def chunks(lst,n): #This will only work if the length of the list is defined
	for i in xrange(0,len(lst),n):
		yield lst[i:i+n]

#
#From: Mining the Social Web p12
def get_rt_source(tweet):
	import re
	rt_patterns = re.compile(r"(RT|via)((?:\b\W*@\w+)+)",re.IGNORECASE)
	return [source.strip() for tup in rt_patterns.findall(tweet) for source in tup if source not in ("RT","via")]

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    from itertools import izip_longest
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

#From http://blog.marcus-brinkmann.de/2011/09/17/a-better-iterator-for-python-couchdb/
def couchdb_pager(db, view_name='_all_docs',
                  startkey=None, startkey_docid=None,
                  endkey=None, endkey_docid=None, bulk=5000):
    # Request one extra row to resume the listing there later.
    options = {'limit': bulk + 1}
    if startkey:
        options['startkey'] = startkey
        if startkey_docid:
            options['startkey_docid'] = startkey_docid
    if endkey:
        options['endkey'] = endkey
        if endkey_docid:
            options['endkey_docid'] = endkey_docid
    done = False
    while not done:
        view = db.view(view_name, **options)
        rows = []
        # If we got a short result (< limit + 1), we know we are done.
        if len(view) <= bulk:
            done = True
            rows = view.rows
        else:
            # Otherwise, continue at the new start position.
            rows = view.rows[:-1]
            last = view.rows[-1]
            options['startkey'] = last.key
            options['startkey_docid'] = last.id

        for row in rows:
            yield row.id
            
def add_random_fields():
	server = couchdb.Server()
	databases = [database for database in server if not database.startswith('_')]
	for database in databases:
		print database
		count = 0
		for document in couchdb_pager(server[database]):
			result = server[database][document]
			if 'results' in result:
				#
				for tweet in result['results']:
					if tweet and 'rand_num' not in tweet:
						tweet['rand_num'] = rand()
						server[database].save(tweet)
			elif 'text' in result and result['text'] and 'rand_num' not in result:
					results['rand_num'] = rand()
					count = count +1
		print count
def select_random(drug,fraction = 0.10):
	import csv
	server = couchdb.Server()
	#databases = [database for database in server if not database.startswith('_')]
	databases = ['toxtweet']
	count = 0
	selected = 0
	for database in databases:
		print database
		with open(database+"_for_rating.csv",'w') as file:
			writer = csv.writer(file)
			for document in couchdb_pager(server[database]):	
				result = server[database][document]
				if 'results' in result:
					for tweet in result['results']:
						if tweet and 'rand_num' in tweet and drug in query:
							if tweet['rand_num'] < fraction:
								row_data = [clean(tweet['text']),tweet['rand_num'],document]
								writer.writerow(row_data)
								selected = selected + 1
				elif 'text' in result and result['rand_num'] and result['rand_num'] < fraction:
					row_data = [clean(result['text']),result['rand_num'],document]
					writer.writerow(row_data)
					selected = selected + 1
			count = count + 1 
			if count%10000==0:
				print "In ",database,selected,' of ',count,'selected'
	wbk.save(database+"_for_rating.xls")
	
def find_all_document_tweets( server=couchdb.Server() ):
    databases = [database for database in server if not database.startswith('_')]
    for database in databases:
        for document in couchdb_pager(server[database]):
            if 'results' in server[database][document]:
                for tweet in server[database][document]['results']:
                    yield database, document, tweet
                    

#Accessory function for NaiveBayesClassifier, perhaps better as a lambda function

def extract_features(tweet,features):
	return {"has(%s)"%feature: (feature in tweet) for feature in features}

def csv_to_corpora(filename): #To allow LSA, LDA from gensim
	import csv
	from gensim import corpora, models, similarities
	from os.path import basename
	as_list=[]
	print 'Loading Tweets'
	with open(filename,'rb') as f:
		reader = csv.reader(f)
		as_list = [row[0].split() for row in reader] #gensim expects a list of words for each document
	print 'Loaded Tweets'
	dictionary = corpora.Dictionary(as_list)
	print 'Made id2word'
	dictionary.save(basename(filename)+'.dict')
	print 'Saved id2word'
	corpus = [dictionary.doc2bow(text) for text in as_list]
	print 'Made corpus'
	corpora.MmCorpus.serialize(basename(filename)+'.mm',corpus) #store for later use
	print 'Saved corpus'
	
	#tfidf = models.TfidfModle(corpus) #LSA and LDA both assume the text has a vector space representation
	#print 'made tfidf'
	
	lda = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word = dictionary, num_topics = 20, 
							update_every=1, chunksize=100000, passes =1) 	
	print 'made LDA'
	
	lda.print_topics(50)
	
#Update the local copy of the CouchDB so that each tweet has a field called rating
def classify_tweets(drug='cocaine'):
	server = couchdb.Server() #default option is local host
	try:
		db = server[drug]
	except: #Can add in specific exceptions later
		print drug,' not in Server'
	
	features,extract_features,classifier = cPickle.load(open(drug+'_classifier.pkl','rb'))
	print 'Unpickled Classifier'		
	
	print 'Loading Relevant Tweets'
	targets = [id for id in db if drug in db[id].keys()]
	print 'Loaded All Relevant Tweets'
	print 'Filtering out tweets that were already rated'
	targets = filter(lambda tweet: 'classification' in tweet.keys() and not tweet['classification'], targets)
	print 'Passing filtered list to classifier'
	for idx, tweet in enumerate(targets):
		#Original queries used just the drug name and no synonyms. 
		#There may be a more efficient Map/Reduce

		#couchdb-python makes all this updating easy
		print tweet
		tweet['classification'] = classifier.classify(extract_features(tweet,features))		
		db.save(tweet)
		print idx/float(len(targets))
	