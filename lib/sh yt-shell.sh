#!/bin/sh

cat drugs | while read line
do
	echo $line
	python ytpy.py $line
	curl http://localhost:5984/_replicate -H 'Content-Type: application/json' -d '{ "source": "yt-cocaine", "target": "https://acharya:marathon221$@acharya.cloudant.com/yt-cocaine" }'
done