from sys import argv
from os import path, makedirs

from switch import *
from YouTubeSearch import *

import utils as tech
import couchdb	

def list_comments(video):										
	return tech.lemmingtize(tech.remove_stopwords(video['comments']))		
					
def save(video,format='txt'):
	for case in switch(format):
		if case('xls'):
			import xlwt
			if len(video['comments']) > 0:
				wbk = xlwt.Workbook()
				sheet = wbk.add_sheet(tech.validate(video['title']))
				bigs = wbk.add_sheet('Bigrams')
				tris = wbk.add_sheet('Trigrams')
				context = wbk.add_sheet('Context')
				for_nlp = tech.flatten(video['comments'][0])
				for idx,comment in enumerate(video['comments'][0]):
					sheet.write(idx,0,' '.join(comment))
				for idx,bigram in enumerate(tech.bigrams(for_nlp,self.term)):
					bigs.write(idx,0,' '.join(bigram))
				for idx,trigram in enumerate(tech.trigrams(for_nlp,self.term)):
					tris.write(idx,0,' '.join(trigram))
				for idx,con in enumerate(tech.context(for_nlp,self.term)):
					context.write(idx,0,' '.join(con))
				wbk.save(tech.validate(video['title'])+'.xls')
			print 'Videos, trigrams, bigrams, and contexts saved to XLS files.'
					#indexing is zero-based, row then column
			break
		if case('txt'):
			if len(video['comments']) > 0:
				#name the file based on the name of the overall search (directory) as well as the video
				with open(path.join(dir_path, tech.validate(video['title'])),'a') as f:
		 			f.write(video['comments'])
			print 'Saved %s as text' % video['title']
			break
		
drug = argv[1]
print 'You are searching YouTube for %s' % drug

#get the directory name based on this search term
dir_path = './%s' %drug
dir_path

#if there is no directory yet for this search term, make a new one
if not path.exists(dir_path):
	print 'Directory /%s does not exist; creating now.' %drug
	makedirs(dir_path)

cocaine_comments = YouTubeSearch(drug)
server  = couchdb.Server()
dbname = 'yt-%s' % drug
for video in cocaine_comments:
	save(video)
	server[dbname].save(video)