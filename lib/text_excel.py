import csv
RESULTS = [
    ['apple','cherry','orange','pineapple','strawberry']
]
resultFile = open("output.csv",'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerow(RESULTS)