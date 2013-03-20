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