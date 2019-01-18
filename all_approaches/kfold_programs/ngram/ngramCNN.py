#python ngramCNN.py --data articles-training-byarticle-20181122.xml --datalabel ground-truth-training-byarticle-20181122.xml > CNN.txt

#python ngramCNN.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml

# MLP for the hyperpartisan problem
import numpy
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
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
	labels_binary = []
	for index in indices:
	#Creating the string version of the index so as to retrieve the article
		a = str(index)
		while (len(a)<7):
			a = '0'+a
		#Retrieving the truth labels using the transformed index as key and building the train/test corpus for the given fold
		label_XML = root_data_label_file.findall("./article[@id= '%s' ]" % a)
		for i in label_XML:
			labels.append(i.attrib['hyperpartisan'])

	for term in labels:
		if term == 'true':
			labels_binary.append(1)
		else:
			labels_binary.append(0)

	return labels_binary

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

for train, test in kfold.split(data):
	print "........... Fold %d ..........." % fold_number
	fold_number = fold_number + 1
	
	train_corpus = build_corpus(train)
	train_labels = build_labels(train)
	
	test_corpus = build_corpus(test)
	test_labels = build_labels(test)
	
	#Generating dummy accuracies for each fold.
	dummy_clf.fit(train_corpus, train_labels)
	dummy_accuracies.append(dummy_clf.score(test_corpus,test_labels) * 100)

	maxlen = 10000

	tokenizer = Tokenizer(num_words=None)
	tokenizer.fit_on_texts(train_corpus)
	vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
	train_vectors = tokenizer.texts_to_sequences(train_corpus)
	train_vectors = pad_sequences(train_vectors, padding='post', maxlen=maxlen)
	test_vectors = tokenizer.texts_to_sequences(test_corpus)
	test_vectors = pad_sequences(test_vectors, padding='post', maxlen=maxlen)

	embedding_dim = 100
	
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
	#model.add(layers.Flatten())
	model.add(layers.Conv1D(128, 5, activation='relu'))
	model.add(layers.GlobalMaxPool1D())
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	history = model.fit(train_vectors, train_labels, epochs=100, verbose=1, validation_data=(test_vectors, test_labels), batch_size=10)
	
	loss, accuracy = model.evaluate(test_vectors, test_labels, verbose=1)
	nn_accuracies.append(accuracy*100)

accuracy_dummy = sum(dummy_accuracies)/len(dummy_accuracies)
accuracy_nn = sum(nn_accuracies)/len(nn_accuracies)

print "Dummy accuracies from each fold:",dummy_accuracies
print "Average dummy accuracy =",round(accuracy_dummy,2),"%"

print "NN accuracies from each fold:",nn_accuracies
print "Average NN accuracy =",round(accuracy_nn,2),"%"