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
Email: ssengupta8@d.umn.edu
'''

'''
This program will do the following:
1. Load the ngram model inside the code i.e. no need to pass it from the command line.
2. Generate the predictions of the input test file passed from the command line according to the SemEval format.

Usage Instructions:
python3 ngram_LR_predict.py -tf <Path to the test file> -o <Path to the folder to which the predictions will be written>

Example Usage:
python3 ngram_LR_predict.py -tf /Users/babun/Desktop/SemEval2k19/data/test/samp_ip/test.xml -o /Users/babun/Desktop/SemEval2k19/data/test/samp_op

python3 ngram_LR_predict.py -tf /Users/babun/Desktop/SemEval2k19/data/custom/test_data/data/test.xml -o /Users/babun/Desktop/SemEval2k19/data/custom/test_data/pred_xml

python3 ngram_LR_predict.py -tf /Users/babun/Desktop/SemEval2k19/data/custom/test_data/data/test.txt -o /Users/babun/Desktop/SemEval2k19/data/custom/test_data/predictions
'''

from sklearn.linear_model import LogisticRegression
from lxml import objectify
from joblib import load
from lxml import etree
import argparse
import errno
import sys
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate predictions for the supplied test data.')

parser.add_argument('-tf','--testfile', metavar='', type=str, help='Path to the test file (XML).' , required = True)
parser.add_argument('-o','--outputpath', metavar='', type=str, help='Path to which the predictions file will be written.', required = True)
parser.add_argument('-tdmn','--tdmname', metavar='', type=str , help='Name of the saved TDM model', default='MyTDM' )
parser.add_argument('-lrmn','--lrmname', metavar='', type=str , help='Name of the saved LR model', default='MyLRM')

args = parser.parse_args()

output_file_path = args.outputpath

qualified_name_of_output_file = output_file_path + "/predictions.txt"

#Reading in the input file.
test_articles = []
test_articles_id = []

if args.testfile.endswith('.xml'):
	test_file = objectify.parse(open(args.testfile,encoding = "utf-8"))
	root_test_file = test_file.getroot()

	print("Reading in the test file:")
	for i in tqdm(root_test_file.getchildren()):
		test_articles.append(' '.join(e for e in i.itertext()))
		test_articles_id.append(i.attrib['id'])

elif args.testfile.endswith('.txt'):
	print("Reading in the test file:")
	articles = open(args.testfile,'r', encoding="utf-8")
	test_articles_total = [ast.literal_eval(line.strip()) for line in articles.readlines()]
	test_articles_id = [x[0] for x in test_articles_total]
	test_articles = [x[1] for x in test_articles_total]

else:
	print("Invalid File Extension. Program is now exiting...")
	exit()

#Loading the TDM model.
ngram_model = load(args.tdmname+".joblib")

#Loading the Classifier.
lr_clf = load(args.lrmname+".joblib")

#Creating the test vectors.
test_vectors = ngram_model.transform(test_articles).toarray()
	
#Making Predictions
print("Making Predictions...")
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
	print("Saving the Predictions...")
	for prediction in lr_predictions:
		f.write(test_articles_id[j] + ' ' + prediction + "\n")
		j = j + 1