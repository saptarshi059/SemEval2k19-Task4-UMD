#python generate_sentiment_vectors.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml 
#python generate_sentiment_vectors.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/articles-training-bypublisher-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml 
#python generate_sentiment_vectors.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml

#python generate_sentiment_vectors.py /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp_train.xml /Users/babun/Desktop/SemEval2k19/data/test/samp_test.xml

from lxml import etree
from lxml import objectify
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import csv
import nltk

train_vectors = {}
test_vectors = {}

train_labels = []
test_labels = []

train_file = objectify.parse(open(sys.argv[1]))
test_file = objectify.parse(open(sys.argv[2]))

root_train_file = train_file.getroot()
root_test_file = test_file.getroot()

#Creating an NLTK sentiment analyzer object
sid = SentimentIntensityAnalyzer()

#Creating the training vectors
for i in root_train_file.getchildren():
	
	sv = [] #Sentiment Score Vector. This vector contains 5 elements resp. - weights of negative, neutral, positive sent. and compound/overall sentiment and subjectivity score.
	article = ' '.join(e for e in i.itertext()).lower()
	blob = TextBlob(article)
	scores = sid.polarity_scores(article)
	
	#Adding the first 4 scores to the sentiment vector.
	for j in scores.keys():
		sv.append(scores[j])
	
	#Appending the subjectivity score.
	sv.append(blob.sentiment[1])
	
	train_vectors[i.attrib['id']] = sv
	
train_vectors = sorted(train_vectors.items(), key=lambda x: x[0]) #Dictionary now converted to list of tuples.

#Creating the test vectors
for i in root_test_file.getchildren():
	
	sv = [] #Sentiment Score Vector. This vector contains 5 elements resp. - weights of negative, neutral, positive sent. and compound/overall sentiment and subjectivity score.
	article = ' '.join(e for e in i.itertext()).lower()
	blob = TextBlob(article)
	scores = sid.polarity_scores(article)
	
	#Adding the first 4 scores to the sentiment vector.
	for j in scores.keys():
		sv.append(scores[j])
	
	#Appending the subjectivity score.
	sv.append(blob.sentiment[1])
	
	test_vectors[i.attrib['id']] = sv
	
test_vectors = sorted(test_vectors.items(), key=lambda x: x[0]) #Dictionary now converted to list of tuples.

train_vectors_numerized = []
test_vectors_numerized = []

for vect in train_vectors:
	train_vectors_numerized.append(vect[1])

for vect in test_vectors:
	test_vectors_numerized.append(vect[1])

with open('train_sent.csv','w') as fp:
	csv.writer(fp).writerows(train_vectors_numerized)

with open('test_sent.csv','w') as fp:
	csv.writer(fp).writerows(test_vectors_numerized)