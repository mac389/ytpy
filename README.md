ytpy
====

A Python API for natural language processing of YouTube

  YouTube provides a large and dynamic data set that may be useful to those researching language. This API makes it 
  easier to analyze linguistic data from YouTube. It has no methods for posting videos or comments. 
  
                                 
                                 Data Acquisition (instantiate YouTubeSearch object)
                                                |
                                                |
                                                |
                                                V
                                      Preprocess (remove stopwords, stem, lemmatize)
                                                |
                                                |
                                                |
                                                V
                                        Analyze (NLTK, Gensim)


Dependencies
------------

 + requests
 + zlib
 + numpy
 + scipy
 + matplotlib
 + couchdb
 + xlwt
 + nltk
 + gensim
 + termcolor
