#These were used for "by_publisher".
#python generate_sentiment_vectors_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/samp_train.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml
#python generate_sentiment_vectors_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/articles-training-bypublisher-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml

#These were used for "by_articles". Not needed anymore.
#python generate_sentiment_vectors_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml 
#python generate_sentiment_vectors_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml
#python generate_sentiment_vectors_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/samp_train.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml

from lxml import etree
from lxml import objectify
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import numpy as np
import nltk

train_file = objectify.parse(open(sys.argv[1]))
test_file = objectify.parse(open(sys.argv[2]))

root_train_file = train_file.getroot()
root_test_file = test_file.getroot()

train_vectors = np.zeros([len(root_train_file.getchildren()),5])
test_vectors = np.zeros([len(root_test_file.getchildren()),5])

#Creating an NLTK sentiment analyzer object
sid = SentimentIntensityAnalyzer()

#Creating the training vectors
article_number = 0
for i in root_train_file.getchildren():

	article = ' '.join(e for e in i.itertext()).lower()
	blob = TextBlob(article)
	scores = sid.polarity_scores(article)
	
	#Adding all the sentiment scores to the train_vectors array.
	s = scores.values() #Sentiment Score Vector. This vector contains 5 elements resp. - weights of negative, neutral, positive sent. and compound/overall sentiment and subjectivity score.
	s.append(blob.sentiment[1])
	train_vectors[article_number] = s 
	article_number = article_number + 1
	
#Creating the test vectors
article_number = 0
for i in root_test_file.getchildren():

	article = ' '.join(e for e in i.itertext()).lower()
	blob = TextBlob(article)
	scores = sid.polarity_scores(article)
	
	#Adding all the sentiment scores to the train_vectors array.
	s = scores.values() #Sentiment Score Vector. This vector contains 5 elements resp. - weights of negative, neutral, positive sent. and compound/overall sentiment and subjectivity score.
	s.append(blob.sentiment[1])
	test_vectors[article_number] = s 
	article_number = article_number + 1

#Saving the vectors to disk as .npy files

np.save('sentiment_vectors_train',train_vectors)
np.save('sentiment_vectors_test',test_vectors)