#python ngram.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 12 > unigram_cutoff12.txt ....best performance for LR with 77%!

#python ngram.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 10 > unigram_cutoff10.txt

#python ngram.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 5 > unigram_cutoff5.txt

#python ngram.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 2 2 --cutoff 12 > bigram_cutoff12.txt

#python ngram.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 2 2 --cutoff 10 > bigram_cutoff10.txt

#python ngram.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 2 2 --cutoff 5 > bigram_cutoff5.txt

from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from sklearn import tree
from sklearn import svm
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from lxml import objectify
from lxml import etree
from tqdm import tqdm
import numpy as np
import argparse
import operator

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

def calc_acc(predictions):
	correctly_classified = 0
	j = 0
	for i in predictions:
		if i == test_labels[j]:
			correctly_classified = correctly_classified + 1
		j = j + 1
	acc = (correctly_classified / len(predictions)) * 100
	return acc

def plot_coefficients(classifier, feature_names, top_features=20):
	coef = classifier.coef_.ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

	# create plot
	plt.figure(figsize=(15, 5))
	colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
	#plt.show()
	plt.savefig('Fold'+str(fold_number)+'.png', bbox_inches='tight')

def features_and_weights(calssifier_name, classifier, feature_names):
	top_features=len(feature_names)
	coef = classifier.coef_.ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
	
	feature_names = np.array(feature_names)
	Weights = {}
	for c in top_coefficients:
		Weights[feature_names[c]] = coef[c]
	
	sorted_Weights = sorted(Weights.items(), key=operator.itemgetter(1), reverse = True)

	f = open(calssifier_name+'_Fold'+str(fold_number)+'_features.txt','w')

	for tup in sorted_Weights:
		f.write(str(tup))
		f.write('\n')

	f.close()

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

dummy_accuracies = []
svm_accuracy_list = []
knn_accuracy_list = []
gnb_accuracy_list = []
dt_accuracy_list = []

lr_accuracy_list = []

#Classifier Object
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
svm = svm.SVC(gamma='auto',kernel='linear')
knn = KNeighborsClassifier(n_neighbors=2)
dt = tree.DecisionTreeClassifier()
gnb = GaussianNB()

lr = LogisticRegression()

# prepare cross validation
kfold = KFold(10, True, 1)
fold_number = 1

for train, test in kfold.split(data):	
	print "........... Fold %d ..........." % fold_number

	training_corpus = build_corpus(train)
	
	train_labels = build_labels(train)

	test_corpus = build_corpus(test)

	test_labels = build_labels(test)

	#Generating dummy accuracies for each fold.
	dummy_clf.fit(training_corpus, train_labels)
	dummy_accuracies.append(dummy_clf.score(test_corpus,test_labels) * 100)

	vectorizer = CountVectorizer(ngram_range = args.ngrange , stop_words=stop_words, binary = args.onehot, analyzer = 'word', token_pattern = r'\b[^\W\d]+\b' )

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
	for key in vectorizer.vocabulary_.keys():
		vectorizer.vocabulary_[key] = j
		j = j + 1

	#Supplying the new vocabulary here.
	print "Retraining the vectorizer with the new vocabulary:"
	vectorizer = CountVectorizer(ngram_range = args.ngrange , stop_words=stop_words , vocabulary=vectorizer.vocabulary_ , binary = args.onehot, analyzer = 'word', token_pattern = r'\b[^\W\d]+\b')
	Trained_Vectors = vectorizer.fit_transform(training_corpus).toarray()
	
	print "Features Used in this iteration were:"
	print vectorizer.vocabulary_.keys()
	print "\n"

	test_vectors = vectorizer.transform(test_corpus).toarray()

	#Training each Classifier
	svm_clf = svm.fit(Trained_Vectors,train_labels)
	knn_clf = knn.fit(Trained_Vectors, train_labels)
	dt_clf = dt.fit(Trained_Vectors, train_labels)
	gnb_clf = gnb.fit(Trained_Vectors, train_labels)
	lr_clf = lr.fit(Trained_Vectors, train_labels)

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

	#plot_coefficients(svm_clf, vectorizer.get_feature_names())
	#features_and_weights('SVM',svm_clf,vectorizer.get_feature_names())
	#features_and_weights('LR',lr_clf,vectorizer.get_feature_names())
	fold_number = fold_number + 1

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