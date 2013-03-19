#create a gigantic tfidf

import couchdb
import cPickle
db_names = ['yt-spice/k2','yt-cocaine','yt-marijuana','yt-science','yt-salvia','yt-coffee','yt-aspirin']
import utils as tech
#retrive all comments in all dbs from 'comments' field and then pile them togethe

'''
server = couchdb.Server()
comments = []
for db_name in db_names:
	print db_name
	db = server[db_name]
	for id in db:
		if 'comments' in db[id].keys():
			comments.append(db[id]['comments']) 
cPickle.dump(comments,open('everything.pkl','wb'))
'''

comments = cPickle.load(open('everything.pkl','rb'))
comments = [comment.split() for comment in comments if comment]
tech.LSA(comments)
'''
for id in db_names:
	if 'comments' in db[id].keys():
		magma.append(db[id]['comments'])
'''
