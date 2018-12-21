#These are used for "by_publisher".
#python generate_ngram_vectors_without_binarizing_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/samp_train.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml

#These were used for "by_articles". Not needed anymore.
#python generate_ngram_vectors_without_binarizing_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml
#python generate_ngram_vectors_without_binarizing_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml
#python generate_ngram_vectors_without_binarizing_numpy.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/samp_train.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml

from lxml import etree
from lxml import objectify
from collections import Counter
from nltk.tokenize import word_tokenize #For getting unigrams.
from nltk.corpus import stopwords
from nltk.util import ngrams #For getting bi & tri grams.
import sys
import numpy as np
import enchant
import re
import nltk

stop_words = set(stopwords.words('english'))
d = enchant.Dict("en_US")

#Creating the xml object/tree
train_file = objectify.parse(open(sys.argv[1]))
test_file = objectify.parse(open(sys.argv[2]))

#To access the root element
root_train_file = train_file.getroot()
root_test_file = test_file.getroot()

train_strings = ''

for i in root_train_file.getchildren():
	train_strings = train_strings + ' '.join(e for e in i.itertext()).lower()

#Normalization & Tokenization (For both Unigram & Bigram Generation)
train_strings = re.sub(r'\W|\d',' ',train_strings)
train_strings_tokenized = nltk.word_tokenize(train_strings) #Generating Unigrams.
train_strings_tokenized_filtered = []
for word in train_strings_tokenized:
	if (word not in stop_words) and (d.check(word) == True):
		train_strings_tokenized_filtered.append(word)

unigrams_wanted = 'yes'
bigrams_wanted = 'n'
trigrams_wanted = 'n'

no_of_unigrams_selected = 0
no_of_bigrams_selected = 0
no_of_trigrams_selected = 0

#Unigram Feature Training
topunigrams_train = []
if unigrams_wanted == 'yes':
	unigram_train_counts = Counter(train_strings_tokenized_filtered)
	unigram_train_counts_sorted = sorted(unigram_train_counts.items(), key=lambda x: x[1], reverse = True)
	no_of_unigrams_selected = 20
	topunigrams_train = unigram_train_counts_sorted[0:no_of_unigrams_selected]

#Bigram Feature Training
topbigrams_train = []
if bigrams_wanted == 'yes':
	bigrams_train = ngrams(train_strings_tokenized_filtered,2)
	bigrams_train_counts = Counter(bigrams_train)
	bigrams_train_counts_sorted = sorted(bigrams_train_counts.items(), key=lambda x: x[1], reverse = True)
	no_of_bigrams_selected = 200
	topbigrams_train = bigrams_train_counts_sorted[0:no_of_bigrams_selected]

#Trigram Feature Training
toptrigrams_train = []
if trigrams_wanted == 'yes':
	trigrams_train = ngrams(train_strings_tokenized_filtered,3)
	trigrams_train_counts = Counter(trigrams_train)
	trigrams_train_counts_sorted = sorted(trigrams_train_counts.items(), key=lambda x: x[1], reverse = True)
	no_of_trigrams_selected = 100
	toptrigrams_train = trigrams_train_counts_sorted[0:no_of_trigrams_selected]

no_of_features = no_of_unigrams_selected + no_of_bigrams_selected + no_of_trigrams_selected

train_vectors = np.zeros([len(root_train_file.getchildren()),no_of_features])
test_vectors = np.zeros([len(root_test_file.getchildren()),no_of_features])

#Generating Final Vectors

#Training Vectors
article_number = 0
for article in root_train_file.getchildren():

	temp = ' '.join(e for e in article.itertext()).lower() #Temporarily storing the article as a string.
	index_number = 0

	for term in topunigrams_train:
		train_vectors[article_number][index_number] = len(re.findall(r'\b'+re.escape(term[0])+r'\b',temp))
		index_number = index_number + 1

	for term in topbigrams_train:
		bigram = ' '.join(t for t in term[0])
		train_vectors[article_number][index_number] = len(re.findall(r'\b'+re.escape(bigram)+r'\b',temp))
		index_number = index_number + 1

	for term in toptrigrams_train:
		trigram = ' '.join(t for t in term[0])
		train_vectors[article_number][index_number] = len(re.findall(r'\b'+re.escape(trigram)+r'\b',temp))
		index_number = index_number + 1	

	article_number = article_number + 1

#Testing Vectors
article_number = 0
for article in root_test_file.getchildren():

	temp = ' '.join(e for e in article.itertext()).lower() #Temporarily storing the article as a string.
	index_number = 0
	
	for term in topunigrams_train:
		test_vectors[article_number][index_number] = len(re.findall(r'\b'+re.escape(term[0])+r'\b',temp))
		index_number = index_number + 1

	for term in topbigrams_train:
		bigram = ' '.join(t for t in term[0])
		test_vectors[article_number][index_number] = len(re.findall(r'\b'+re.escape(bigram)+r'\b',temp))
		index_number = index_number + 1

	for term in toptrigrams_train:
		trigram = ' '.join(t for t in term[0])
		test_vectors[article_number][index_number] = len(re.findall(r'\b'+re.escape(trigram)+r'\b',temp))
		index_number = index_number + 1

	article_number = article_number + 1

np.save('ngram_training_vectors',train_vectors)
np.save('ngram_testing_vectors',test_vectors)