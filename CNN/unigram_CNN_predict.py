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
This program will do the following:
1. Load the CNN model inside the code i.e. no need to pass it from the command line.
2. Generate the predictions for the input test file passed from the command line according to the SemEval format.

Usage Instructions:
python unigram_CNN_predict.py -tf <Path to the folder holding the test file> -o <Path to the folder to which the predictions will be written>

Example Usage:
python unigram_CNN_predict.py -tf /Users/babun/Desktop/SemEval2k19/data/test/samp_ip -o /Users/babun/Desktop/SemEval2k19/data/test/samp_op

python unigram_CNN_predict.py -tf ~/Desktop/semeval2k19/data/train_data/train_byarticle/custom/test_data/data -o ~/Desktop/semeval2k19/data/train_data/train_byarticle/predictions_custom
'''

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import load_model
from lxml import objectify
from keras import layers
from joblib import load
from lxml import etree
import argparse
import errno
import sys
import os
import ast

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-tf','--testfile', metavar='', type=str, help='Path to the test file(.xml/.txt)' , required = True)
parser.add_argument('-o','--outputpath', metavar='', type=str, help='Path to the folder in which the predictions file(.txt) will be saved.', required = True)
parser.add_argument('-mn','--modelname', metavar='', type=str, help='Name of the saved CNN model', default='myCNN')
parser.add_argument('-tn','--tokenizername', metavar='', type=str, help='Name of the saved tokenizer', default='mytokenizer')

args = parser.parse_args()

output_file_path = args.outputpath

qualified_name_of_output_file = output_file_path + "/predictions.txt"

#Reading in the input file.
test_articles = []
test_articles_id = []

if args.testfile.endswith('.xml'):
	test_file = objectify.parse(open(args.testfile,encoding = "utf-8"))
	root_test_file = test_file.getroot()

	for i in root_test_file.getchildren():
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

#Loading the CNN model.
model = load_model(args.modelname)

#Loading the Classifier.
tokenizer = load(args.tokenizername)

#Creating the test vectors.
maxlen = 10000
test_vectors = tokenizer.texts_to_sequences(test_articles)
test_vectors = pad_sequences(test_vectors, padding='post', maxlen=maxlen)
	
#Making Predictions
CNN_predictions = model.predict_classes(test_vectors)

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
	for prediction in CNN_predictions:
		if prediction[0] == 0:
			f.write(test_articles_id[j] + ' ' + 'false' + '\n')
		else:
			f.write(test_articles_id[j] + ' ' + 'true' + '\n')
		j = j + 1