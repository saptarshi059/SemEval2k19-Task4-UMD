'''
Code Usage Instructions:

Run the program using the following command line instructions:

python ngram_train.py --train <Path to training file> --train_label <Path to training labels> --ngram <Type of n-grams wanted> --topn <Top N n-gram features wanted> --onehot <Whether or not you want 1 hot encoding (False) or regular count based vectors (True)> --op1 <Name to save the training model as> --op2 <Name to save the training labels as>

Example Usage:

python ngram_train.py --train /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp/samp_train.xml --train_label /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp_train_label.xml --ngram 2 --topn 10 --onehot False --op1 ngram_model --op2 model_labels
'''

#python ngram_train.py --train /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp/samp_train.xml --train_label /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp_train_label.xml --ngram 2 --topn 10 --onehot False --op1 ngram_model --op2 model_labels
#python ngram_train.py --train /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --train_label /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngram 1 --topn 100 --onehot False --op1 ngram_model --op2 model_labels

from sklearn.feature_extraction.text import CountVectorizer
from lxml import etree
from lxml import objectify
from nltk.corpus import stopwords
from tqdm import tqdm
import sys
import nltk
import ast
import numpy as np
from joblib import dump

stop_words_list = set(stopwords.words('english'))

#Creating the xml object/tree
train_file = objectify.parse(open(sys.argv[2]))
train_label_file = objectify.parse(open(sys.argv[4]))

#To access the root element
root_train_file = train_file.getroot()
root_train_label_file = train_label_file.getroot()

training_corpus = []
training_labels = np.array([],dtype=object)

print "Reading in the training corpus:"
for i in tqdm(root_train_file.getchildren()):
	training_corpus.append(' '.join(e for e in i.itertext()))

vectorizer = CountVectorizer(analyzer = 'word', stop_words = stop_words_list , ngram_range = (int(sys.argv[6]),int(sys.argv[6])) , 
							max_features = None if ast.literal_eval(sys.argv[8]) is None else int(sys.argv[8]) , binary = ast.literal_eval(sys.argv[10]))

print "Building the model:"
vectors = vectorizer.fit_transform(training_corpus)

print "\nThe features selected were:"
print vectorizer.vocabulary_.keys()

for row in tqdm(root_train_label_file.getchildren()):
	training_labels = np.append(training_labels, row.attrib['hyperpartisan'])

np.save(sys.argv[14], training_labels)
np.save('training_vectors', vectors.toarray())
dump(vectorizer, sys.argv[12]+'.joblib')
print "\nThe Ngram Model was saved..."