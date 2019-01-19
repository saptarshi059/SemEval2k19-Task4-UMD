'''
This program will do the following:
1. Load the ngram model inside the code i.e. no need to pass it from the command line.
2. Generate the predictions of the input test file passed from the command line according to the SemEval format.

Usage Instructions:
python ngram_LR_predict.py -tf <Path to the folder holding the test file> -o <Path to the folder to which the predictions will be written>

Example Usage:
python ngram_LR_predict.py -tf /Users/babun/Desktop/SemEval2k19/data/test/samp_ip -o /Users/babun/Desktop/SemEval2k19/data/test/samp_op
'''

from sklearn.linear_model import LogisticRegression
from lxml import objectify
from joblib import load
from lxml import etree
import argparse
import errno
import sys
import os

parser = argparse.ArgumentParser(description='Generate predictions for the supplied test data.')

parser.add_argument('-tf','--testfile', metavar='', type=str, help='Path to the test file (XML).' , required = True)
parser.add_argument('-o','--outputpath', metavar='', type=str, help='Path to which the predictions file will be written.', required = True)

args = parser.parse_args()

input_file_path = args.testfile
output_file_path = args.outputpath

qualified_name_of_output_file = output_file_path+"/predictions.txt"

#Reading in the imput file.
for filename in os.listdir(input_file_path):
	if filename.endswith('.xml'):
		fullname = os.path.join(input_file_path,filename)
		test_file = objectify.parse(fullname)

root_test_file = test_file.getroot()

test_articles = []
test_articles_id = []

for i in root_test_file.getchildren():
	test_articles.append(' '.join(e for e in i.itertext()))
	test_articles_id.append(i.attrib['id'])

#Loading the TDM model.
ngram_model = load('TDM.joblib')

#Loading the Classifier.
lr_clf = load('LR.joblib')

#Creating the test vectors.
test_vectors = ngram_model.transform(test_articles).toarray()
	
#Making Predictions
lr_predictions = lr_clf.predict(test_vectors)

#If the directory doesn't exist, create a directory.
if not os.path.exists(os.path.dirname(qualified_name_of_output_file)):
    try:
        os.makedirs(os.path.dirname(qualified_name_of_output_file))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#Writing predictions to the output file in the output directory.
with open(qualified_name_of_output_file, "w") as f:
	j = 0 
	for prediction in lr_predictions:
		f.write(test_articles_id[j] + ' ' + prediction + '\n')
		j = j + 1