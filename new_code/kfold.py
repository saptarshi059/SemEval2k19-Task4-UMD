#python kfold.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 10
from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from lxml import objectify
from sklearn import svm
from lxml import etree
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-t','--data', metavar='', type=str, help='Path to training (or data file, since we are using KFold cross validation) file (XML).' , required = True)
parser.add_argument('-tl','--datalabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)
parser.add_argument('-ngr', '--ngrange' ,metavar='', nargs=2 ,type=int, help='Types of ngrams wanted as features: ex. for unigrams enter 1 1, unigrams and bigrams enter 1 2 etc.', required = True)
parser.add_argument('-c','--cutoff', metavar='', type=int, help='Select only those features which have frequency higher than this value.', required = True)
parser.add_argument('-oh','--onehot' , action='store_true' ,help='Whether or not you want the vectors to be one hot encoded. If yes, set/include this argument in the command line argument list else leave it.')

args = parser.parse_args()

def build_corpus(indices):
	corpus = []
	for index in indices:
	#Creating the string version of the index so as to retrieve the article
		a = str(index)
		while (len(a)<7):
			a = '0'+a
		#Retrieving the article using the transformed index as key and building the train/test corpus for the given fold
		article_XML = root_data_file.findall("./article[@id= '%s' ]" % a)
		for i in article_XML:
			corpus.append(' '.join(e for e in i.itertext()))
	return corpus

def build_labels(indices):
	labels = []
	for index in indices:
	#Creating the string version of the index so as to retrieve the article
		a = str(index)
		while (len(a)<7):
			a = '0'+a
		#Retrieving the truth labels using the transformed index as key and building the train/test corpus for the given fold
		label_XML = root_data_label_file.findall("./article[@id= '%s' ]" % a)
		for i in label_XML:
			labels.append(i.attrib['hyperpartisan'])
	return labels

#Creating the xml object/tree
data_file = objectify.parse(open(args.data))
data_label_file = objectify.parse(open(args.datalabel))

#To access the root element
root_data_file = data_file.getroot()
root_data_label_file = data_label_file.getroot()

data = []
data_labels = []

print "Reading in the training corpus:"
for i in tqdm(root_data_file.getchildren()):
	data.append(' '.join(e for e in i.itertext()))

print "Reading in the training label file:"
for row in tqdm(root_data_label_file.getchildren()):
	data_labels.append(row.attrib['hyperpartisan'])

stop_words = set(stopwords.words('english'))

svm_accuracy_list = []

#Classifier Object
svm = svm.SVC(gamma='auto',kernel='rbf')

# prepare cross validation
kfold = KFold(10, True, 1)

for train, test in kfold.split(data):	
	training_corpus = build_corpus(train)
	
	train_labels = build_labels(train)

	test_corpus = build_corpus(test)

	test_labels = build_labels(test)

	vectorizer = CountVectorizer(ngram_range = args.ngrange , stop_words=stop_words, binary = args.onehot)

	vectors = vectorizer.fit_transform(training_corpus).toarray()

	sums = vectors.sum(axis=0)

	j = 0
	print "Cleaning up the TDM according to the supplied cutoff value:"
	for i in tqdm(vectorizer.vocabulary_.items()):
		if sums[i[1]] < args.cutoff:
			del vectorizer.vocabulary_[i[0]]
		j = j + 1

	#We have to do this step because the test vectors will have to be transformed on a new TDM i.e. the one which reflects the dropped features.
	j = 0
	print "Retraining the vectorizer with the new vocabulary:"
	for key in vectorizer.vocabulary_.keys():
		vectorizer.vocabulary_[key] = j
		j = j + 1

	#Supplying the new vocabulary here.
	vectorizer = CountVectorizer(ngram_range = args.ngrange , stop_words=stop_words , vocabulary=vectorizer.vocabulary_ , binary = args.onehot)
	Trained_Vectors = vectorizer.fit_transform(training_corpus).toarray()
	
	print "Features Used in this iteration were:"
	print vectorizer.vocabulary_.keys()
	print "\n"

	test_vectors = vectorizer.transform(test_corpus).toarray()

	#Training the Classifier
	svm_clf = svm.fit(Trained_Vectors,train_labels)

	#Making Predictions
	svm_predictions = svm_clf.predict(test_vectors)

	correctly_classified = 0
	j = 0
	for i in svm_predictions:
		if i == test_labels[j]:
			correctly_classified = correctly_classified + 1
		j = j + 1

	acc = (correctly_classified / len(svm_predictions)) * 100

	svm_accuracy_list.append(acc)

accuracy = sum(svm_accuracy_list)/len(svm_accuracy_list)

print "Accuracies from each fold:",svm_accuracy_list
print "Average accuracy of the classifier =",round(accuracy,2),"%"