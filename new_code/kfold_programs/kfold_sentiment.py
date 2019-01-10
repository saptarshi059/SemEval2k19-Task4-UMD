#python kfold_sentiment.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml > sentiment.txt
from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from lxml import objectify
from lxml import etree
from tqdm import tqdm
import numpy as np
import argparse
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-t','--data', metavar='', type=str, help='Path to training (or data file, since we are using KFold cross validation) file (XML).' , required = True)
parser.add_argument('-tl','--datalabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)

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

def build_vectors(indices):
	vectors = np.zeros([len(indices),5])
	j = 0
	for index in indices:
	#Creating the string version of the index so as to retrieve the article.
		a = str(index)
		while (len(a)<7):
			a = '0'+a
		#Retrieving the required article using the transformed index as key.
		article_XML = root_data_file.findall("./article[@id= '%s' ]" % a)
		for i in article_XML:			
			article = ' '.join(e for e in i.itertext()).lower()
			blob = TextBlob(article)
			scores = sid.polarity_scores(article)
			s = scores.values() #Sentiment Score Vector. This vector contains 5 elements resp. - weights of negative, 							neutral, positive sent. and compound/overall sentiment and subjectivity score.
			s.append(blob.sentiment[1])
			vectors[j] = s
		j = j + 1
	return vectors

def calc_acc(predictions):
	correctly_classified = 0
	j = 0
	for i in predictions:
		if i == test_labels[j]:
			correctly_classified = correctly_classified + 1
		j = j + 1
	acc = (correctly_classified / len(predictions)) * 100
	return acc

#Creating an NLTK sentiment analyzer object
sid = SentimentIntensityAnalyzer()

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

#Classifier Object
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
svm = svm.SVC(gamma='scale',kernel='linear')
knn = KNeighborsClassifier(n_neighbors=3)
dt = tree.DecisionTreeClassifier()
gnb = GaussianNB()
lr = LogisticRegression()

dummy_accuracies = []
svm_accuracy_list = []
knn_accuracy_list = []
gnb_accuracy_list = []
dt_accuracy_list = []
lr_accuracy_list = []

# prepare cross validation
kfold = KFold(10, True, 1)
fold_number = 1

for train, test in kfold.split(data):	
	print "........... Fold %d ..........." % fold_number
	fold_number = fold_number + 1
	
	train_vectors = build_vectors(train)
	training_corpus = build_corpus(train)
	train_labels = build_labels(train)
	
	test_vectors = build_vectors(test)
	test_corpus = build_corpus(test)
	test_labels = build_labels(test)

	#Generating dummy accuracies for each fold.
	dummy_clf.fit(training_corpus, train_labels)
	dummy_accuracies.append(dummy_clf.score(test_corpus,test_labels) * 100)

	#Training each Classifier
	svm_clf = svm.fit(train_vectors,train_labels)
	knn_clf = knn.fit(train_vectors, train_labels)
	dt_clf = dt.fit(train_vectors, train_labels)
	gnb_clf = gnb.fit(train_vectors, train_labels)
	lr_clf = lr.fit(train_vectors, train_labels)

	#Making Predictions
	svm_predictions = svm_clf.predict(test_vectors)
	knn_predictions = knn_clf.predict(test_vectors)
	dt_predictions = dt_clf.predict(test_vectors)
	gnb_predictions = gnb_clf.predict(test_vectors)
	lr_predictions = lr_clf.predict(test_vectors)

	svm_accuracy_list.append(calc_acc(svm_predictions))
	knn_accuracy_list.append(calc_acc(knn_predictions))
	dt_accuracy_list.append(calc_acc(dt_predictions))
	gnb_accuracy_list.append(calc_acc(gnb_predictions))
	lr_accuracy_list.append(calc_acc(lr_predictions))

accuracy_dummy = sum(dummy_accuracies)/len(dummy_accuracies)
accuracy_svm = sum(svm_accuracy_list)/len(svm_accuracy_list)
accuracy_knn = sum(knn_accuracy_list)/len(knn_accuracy_list)
accuracy_dt = sum(dt_accuracy_list)/len(dt_accuracy_list)
accuracy_gnb = sum(gnb_accuracy_list)/len(gnb_accuracy_list)
accuracy_lr = sum(lr_accuracy_list)/len(lr_accuracy_list)

print "Dummy accuracies from each fold:",dummy_accuracies
print "Average dummy accuracy =",round(accuracy_dummy,2),"%"

print "SVM accuracies from each fold:",svm_accuracy_list
print "Average SVM accuracy of the classifier =",round(accuracy_svm,2),"%"

print "KNN accuracies from each fold:",knn_accuracy_list
print "Average KNN accuracy of the classifier =",round(accuracy_knn,2),"%"

print "DT accuracies from each fold:",dt_accuracy_list
print "Average DT accuracy of the classifier =",round(accuracy_dt,2),"%"

print "GNB accuracies from each fold:",gnb_accuracy_list
print "Average GNB accuracy of the classifier =",round(accuracy_gnb,2),"%"

print "LR accuracies from each fold:",lr_accuracy_list
print "Average LR accuracy of the classifier =",round(accuracy_lr,2),"%"