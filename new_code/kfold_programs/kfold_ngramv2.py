#python kfold_ngramv2.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 12

#python kfold_ngramv2.py --data articles-training-byarticle-20181122.xml --datalabel ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 12

# MLP for the hyperpartisan problem
import numpy
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from lxml import objectify
from lxml import etree
from tqdm import tqdm
from nltk.corpus import stopwords
import argparse
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-t','--data', metavar='', type=str, help='Path to training (or data file, since we are using KFold cross validation) file (XML).' , required = True)
parser.add_argument('-tl','--datalabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)
parser.add_argument('-ngr', '--ngrange' ,metavar='', nargs=2 ,type=int, help='Types of ngrams wanted as features: ex. for unigrams enter 1 1, unigrams and bigrams enter 1 2 etc.', required = True)
parser.add_argument('-c','--cutoff', metavar='', type=int, help='Select only those features which have frequency higher than this value.', required = True)
parser.add_argument('-oh','--onehot' , action='store_true' ,help='Whether or not you want the vectors to be one hot encoded. If yes, set/include this argument in the command line argument list else leave it.')

args = parser.parse_args()

stop_words = set(stopwords.words('english'))

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

dummy_accuracies = []
nn_accuracies = []

#Classifier Object
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)

# prepare cross validation
kfold = KFold(10, True, 1)
fold_number = 1

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

for train, test in kfold.split(data):
	print "........... Fold %d ..........." % fold_number
	fold_number = fold_number + 1
	
	training_corpus = build_corpus(train)
	train_labels = build_labels(train)
	test_corpus = build_corpus(test)
	test_labels = build_labels(test)

	#Generating dummy accuracies for each fold.
	dummy_clf.fit(training_corpus, train_labels)
	dummy_accuracies.append(dummy_clf.score(test_corpus,test_labels) * 100)

	tk = Tokenizer()
	tk.fit_on_texts(train_labels)
	train_labels = numpy.array(tk.texts_to_sequences(train_labels))
	test_labels = numpy.array(tk.texts_to_sequences(test_labels))

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
	train_vectors = vectorizer.fit_transform(training_corpus).toarray()

	print "Features Used in this iteration were:"
	print vectorizer.vocabulary_.keys()
	print "\n"	

	test_vectors = vectorizer.transform(test_corpus).toarray()

	input_dim = train_vectors.shape[1]  # Number of features
	
	# create the model
	model = Sequential()
	model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Fit the model
	model.fit(train_vectors, train_labels, validation_data=(test_vectors, test_labels), epochs=100, batch_size=10, verbose=1)
	
	# Final evaluation of the model
	scores = model.evaluate(test_vectors, test_labels, verbose=False)
	nn_accuracies.append(scores[1]*100)

accuracy_dummy = sum(dummy_accuracies)/len(dummy_accuracies)
accuracy_nn = sum(nn_accuracies)/len(nn_accuracies)

print "Dummy accuracies from each fold:",dummy_accuracies
print "Average dummy accuracy =",round(accuracy_dummy,2),"%"

print "NN accuracies from each fold:",nn_accuracies
print "Average NN accuracy =",round(accuracy_nn,2),"%"