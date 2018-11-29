#python generate_vectors.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml
#python generate_vectors.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/articles-training-bypublisher-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml 
#python generate_vectors.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/articles-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml

#python generate_vectors.py /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp_train.xml /Users/babun/Desktop/SemEval2k19/data/test/samp_test.xml

from lxml import etree
from lxml import objectify
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
import csv
import re
import nltk

stop_words = set(stopwords.words('english'))

#Creating the xml object/tree
train_file = objectify.parse(open(sys.argv[1]))
test_file = objectify.parse(open(sys.argv[2]))

#To access the root element
root_train_file = train_file.getroot()
root_test_file = test_file.getroot()


'''
#Using presence of images as features
train_vectors = {}
test_vectors = {}

train_vectors_numerized = []
test_vectors_numerized = []

for article in root_train_file.getchildren():
	if re.search(r'images*',' '.join(e for e in article.itertext()).lower().encode('utf-8')):
		train_vectors[article.attrib['id']] = 1
	else:
		train_vectors[article.attrib['id']] = 0

for article in root_test_file.getchildren():
	if re.search(r'images*',' '.join(e for e in article.itertext()).lower().encode('utf-8')):
		test_vectors[article.attrib['id']] = 1
	else:
		test_vectors[article.attrib['id']] = 0

for i in sorted(train_vectors.items(), key=lambda x:x[0]):
	train_vectors_numerized.append([i[1]])

for i in sorted(test_vectors.items(), key=lambda x:x[0]):
	test_vectors_numerized.append([i[1]])
'''

#////.....Unigram Feature Training.....////#

#Normalization & Tokenization
train_strings = ''
test_strings = ''

for i in root_train_file.getchildren():
	train_strings = train_strings + ' '.join(e for e in i.itertext()).lower()

for i in root_test_file.getchildren():
	test_strings = test_strings + ' '.join(e for e in i.itertext()).lower().encode('utf-8')

train_strings = re.sub(r'\W|\d',' ',train_strings)
train_strings_tokenized = nltk.word_tokenize(train_strings)

train_vectors = {}
test_vectors = {}

vocab_train = Counter(train_strings_tokenized)

for term in vocab_train.keys():
	#Filtering out stop words
	if term in stop_words:
		del vocab_train[term]

	#Deleting those terms which occur less than 5 times
	if vocab_train[term] < 5:
		del vocab_train[term]

vocab_train_sorted = sorted(vocab_train.items(), key=lambda x: x[1], reverse = True)
topunigrams_train = vocab_train_sorted[0:995] #Modifiy this list to change the number of unigrams to select.

print len(topunigrams_train)


for article in root_train_file.getchildren():
	vect = {}
	temp = ' '.join(e for e in article.itertext()).lower().encode('utf-8') #Temporarily storing the training string.

	for term in topunigrams_train:
		n = len(re.findall(re.escape(term[0].encode('utf-8')),temp))
		vect[term[0].encode('utf-8')] = n

	train_vectors[article.attrib['id']] = vect

for article in root_test_file.getchildren():
	vect = {}
	temp = ' '.join(e for e in article.itertext()).lower().encode('utf-8') #Temporarily storing the test string.

	for term in topunigrams_train:
		n = len(re.findall(re.escape(term[0].encode('utf-8')),temp))
		vect[term[0].encode('utf-8')] = n

	test_vectors[article.attrib['id']] = vect

train_vectors = sorted(train_vectors.items(), key=lambda x: x[0])
test_vectors = sorted(test_vectors.items(), key=lambda x: x[0])

train_vectors_numerized_final = []
test_vectors_numerized_final = []

j = 0
for vect in train_vectors:
	#train_vectors_numerized_final.append(vect[1].values() + train_vectors_numerized[j])
	train_vectors_numerized_final.append(vect[1].values())
	j = j + 1

j = 0
for vect in test_vectors:
	#test_vectors_numerized_final.append(vect[1].values() + test_vectors_numerized[j])
	test_vectors_numerized_final.append(vect[1].values())
	j = j + 1

with open('train_unigram.csv','w') as fp:
	csv.writer(fp).writerows(train_vectors_numerized_final)

with open('test_unigram.csv','w') as fp:
	csv.writer(fp).writerows(test_vectors_numerized_final)