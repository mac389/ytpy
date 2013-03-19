import csv
RESULTS = [
    ['apple','cherry'], ['test', 'testb']
]
resultFile = open("output.csv",'wb')
wr = csv.writer(resultFile, dialect='excel')
for comment_list in RESULTS: 
	for comment in comment_list: 
		wr.writerow([comment])
