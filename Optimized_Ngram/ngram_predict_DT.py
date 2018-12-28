'''
This program will do the following:
1. Load the ngram model inside the code i.e. no need to pass it from the command line.
2. Generate the predictions of the input test file passed from the command line according to SemEval format.
'''

#python ngram_predict_DT.py -i /Users/babun/Desktop/SemEval2k19/data/test/samp_ip -o /Users/babun/Desktop/SemEval2k19/data/test/samp_op

from lxml import etree
from lxml import objectify
from joblib import load
from sklearn import tree
import sys
import os
import errno
import numpy as np

input_file_path = sys.argv[2]
output_file_path = sys.argv[4]

qualified_name_of_output_file = output_file_path+"/predictions.txt"

#Reading in the imput file.
for filename in os.listdir(input_file_path):
	if filename.endswith('.xml'):
		fullname = os.path.join(input_file_path,filename)
		test_file = objectify.parse(fullname)

root_test_file = test_file.getroot()

#Loading the model.
ngram_model = load('ngram_model.joblib')

#Loading training vectors.
train_vectors = np.load('training_vectors.npy')

#Loading training labels.
train_labels = np.load('model_labels.npy')

#If the directory doesn't exist, create a directory.
if not os.path.exists(os.path.dirname(qualified_name_of_output_file)):
    try:
        os.makedirs(os.path.dirname(qualified_name_of_output_file))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

test_articles = []
test_articles_labels = []

for i in root_test_file.getchildren():
	test_articles.append(' '.join(e for e in i.itertext()))
	test_articles_labels.append(i.attrib['id'])

#Writing predictions to the output file in the output directory.
with open(qualified_name_of_output_file, "w") as f:
	test_vectors = ngram_model.transform(test_articles).toarray()
	
	#Classifier Object
	dt = tree.DecisionTreeClassifier()

	#Training the Classifier
	dt_clf = dt.fit(train_vectors,train_labels)

	#Making Predictions
	dt_predictions = dt_clf.predict(test_vectors)

	j = 0 

	for prediction in dt_predictions:
		f.write(test_articles_labels[j] + ' ' + prediction + '\n')
		j = j + 1