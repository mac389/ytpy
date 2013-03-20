import os
import requests

import utils as tech

from os import path
from time import sleep
from numpy.random import random
from termcolor import colored,cprint


class YouTubeSearch(object):
	def __init__(self,term,exclude_videos=True): 
		self.term = term
		self.query_url = 'https://gdata.youtube.com/feeds/api/videos'
		self.query_params = {
								'q' : term,
								'alt':'json',
								'orderby':'relevance',
								'v':'2',
								'max-results':50}
		self.video_params = {
								'alt':'json',
								'max-results':50}

		self.exopath = '/Volumes/My Book/Dropbox/ToxTweet/Software/APIs/ytpy/constants'
		self.video_stopwords = ''.join(open(self.exopath+'/'+'entities','rb').readlines()) if exclude_videos else None
		
	def __repr__(self):
		return self.__class__.__name__+repr((self.term))					

	def _do_requests(self):
		#Maximum number of videos per feed is 1,000
		#Max number of videos per page is 50
		res = []
		for start in xrange(1,1000,50):
			self.query_params['start-index'] = start
			interim = requests.get(self.query_url, params = self.query_params).json()
			res.append(interim)
			if len(interim) < 50:
				break
			del interim
		return res
	
	def __iter__(self):
		for batch in self._do_requests():
			for video in batch.get('feed').get('entry'):
				result = {}
				result['title'] = video.get('title').get('$t')
				
				comments =  video['media$group']['media$description']['$t']
				print '-----'
				if tech.should_exclude_video(comments,self.video_stopwords):
					cprint ('\nSkipping %s. It has too many stopwords.' % result['title'],'red')
					del result
					print '-----'
					continue
				
				result['url'] = video.get('link')[0].get('href')
				
				id = video.get('id').get('$t').split(':')[-1]
				video_query = self.query_url+'/'+id+'/comments'
				comments = []
				raw = []
				#iterate over the range of comments that YT allows 
				for i in xrange(1,950,50):
					self.video_params['start-index'] = i
					#chill for a sec
					sleep(1)
					try:
						interim = requests.get(video_query, params = self.video_params).json()
						if 'entry' in interim['feed']:
							raw.extend([comment['content']['$t'] 
											for comment in interim['feed']['entry'] 
											if 'entry' in interim['feed']])
							comments.extend([tech.clean(comment['content']['$t']).split() 
											for comment in interim['feed']['entry'] 
											if 'entry' in interim['feed']])
					except ValueError:
						print "Couldn't get 50 comments starting from %d. Moving on." % i
						continue
				comments = [item for sublist in comments for item in sublist]
				result['comments'] = ' '.join(raw)
				result['comments-as-list'] = raw	
				#result['comments'] = ' '.join(tech.lemmingtize(tech.remove_stopwords(comments)))
				result['random'] = random()
				yield result