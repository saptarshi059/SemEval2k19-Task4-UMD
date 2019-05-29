'''
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Saptarshi Sengupta
Email: sengu059@d.umn.edu
'''

'''
Program Name: d2v.py
Author: SAPTARSHI SENGUPTA
Major: Computer Science / 1st Year / Graduate Student (MS) / University of Minnesota Duluth

Program Details: The d2v program implements our Doc2Vec model for the task. After splitting the supplied data set into training and testing folds (cross-validation), articles in both get vectorized. Each article in the training corpus was converted into a "Tagged Document" which was used to train the document model. Finally, the training vectors were used to train the document model and predictions were made by it on the "inferred" test vectors i.e. articles which are not in the training corpus but whose representation is learned by the model according to the data on which it has been trained.

Code Usage: In order to use this program -
				* A user must have the Python programming language compiler installed.
				
				* The user should type the following command in either command prompt or linux shell i.e. 
				  				python d2v.py -d <path to training data file> -dl <path to training data's label file>

				* An example usage would be of the form: python d2v.py --data /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --datalabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml

Settings Used: vector size = 100.
			   alpha = 0.025.
			   epochs (iterations for training the model) = 100.
			   min_count = 5 i.e. Ignore those terms which have a frequency less than this value.

Program Algorithm: The program has the following underlying logic -
				 
				//PreProcessing
	
				Articles in the training and testing fold were converted to lowercase. Articles in the training corpus was also converted to a "Tagged Document" (explained in point 3 below).

				//Main Program Logic

				1. At first, the XML training file is parsed so as to retrieve each article which in turn is stored in memory in a python list. The same takes place for the training label's file.

				2. A 10 fold training-testing split is done on the data.

				3. The training and testing corpora are built for the current fold. Articles in the training fold (corpus) are associated with their respective 'tags' which in this case is their hyperpartisanship attribute (whether they are hyp. or not) i.e. True/False.

				4. The document model was trained on the training corpus and then tested on the test corpus, by converting each article in the test fold to an 'inferred vector'.

		    	5. Finally, the accuracy of classification is computed for the current fold.
'''
from __future__ import division
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords
from lxml import objectify
from lxml import etree
from tqdm import tqdm
import numpy as np
import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-d','--data', metavar='', type=str, help='Path to training (or data file, since we are using KFold cross validation) file (XML).' , required = True)
parser.add_argument('-dl','--datalabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)

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

#Creating a Dummy Classifier
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)

d2v_accuracy_list = []
dummy_accuracies = []
# prepare cross validation
kfold = KFold(10, True, 1)
fold_number = 1

for train, test in kfold.split(data):	
	print "........... Fold %d ..........." % fold_number
	fold_number = fold_number + 1
	
	training_corpus = build_corpus(train)
	train_labels = build_labels(train)
	test_corpus = build_corpus(test)
	test_labels = build_labels(test)

	dummy_clf.fit(training_corpus, train_labels)
	dummy_accuracies.append(dummy_clf.score(test_corpus,test_labels) * 100)

	#Assigning hyperpartisan (true or false) tags to each document.
	print "Creating the Tagged version of the training_corpus"
	tagged_data = []
	j = 0
	for i in tqdm(training_corpus):
	    tagged_data.append(TaggedDocument(i.lower(),tags=[train_labels[j]]))
	    j = j + 1

	#I'll make these command line args later
	vec_size = 100
	alpha = 0.025

	model = Doc2Vec(tagged_data, vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=5, dm =1, epochs = 100)

	predictions = []

	print "Generating predictions for the test corpus of this fold"
	for i in tqdm(test_corpus):
		v1 = model.infer_vector(i.lower())
		similar_doc = model.docvecs.most_similar([v1])
		if similar_doc[0][1] > similar_doc[1][1]:
			predictions.append(similar_doc[0][0])
		else:
			predictions.append(similar_doc[1][0])

	correctly_classified = 0
	j = 0

	print "Performing accuracy calculations"
	for i in tqdm(predictions):
		if i == test_labels[j]:
			correctly_classified = correctly_classified + 1
		j = j + 1

	acc = (correctly_classified/len(predictions)) * 100

	d2v_accuracy_list.append(acc)

accuracy_dummy = sum(dummy_accuracies)/len(dummy_accuracies)
accuracy_main = sum(d2v_accuracy_list)/len(d2v_accuracy_list)

print "Dummy accuracies from each fold:",dummy_accuracies
print "Average dummy accuracy =",round(accuracy_dummy,2),"%"

print "Main accuracies from each fold:",d2v_accuracy_list
print "Main average accuracy of the classifier =",round(accuracy_main,2),"%"