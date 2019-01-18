'''
This program will do the following:
1. Load the doc2vec model inside the code i.e. no need to pass it from the command line.
2. Generate the predictions of the input test file passed from the command line according to SemEval format.
'''

#python docvec_approach_forTIRA.py -i /Users/babun/Desktop/SemEval2k19/data/test/samp_ip -o /Users/babun/Desktop/SemEval2k19/data/test/samp_op

from __future__ import division
from gensim.models.doc2vec import Doc2Vec
from lxml import etree
from lxml import objectify
import sys
import os
import errno

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
model = Doc2Vec.load('d2v.model')

#If the directory doesn't exist, create a directory.
if not os.path.exists(os.path.dirname(qualified_name_of_output_file)):
    try:
        os.makedirs(os.path.dirname(qualified_name_of_output_file))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#Writing predictions to the output file in the output directory.
with open(qualified_name_of_output_file, "w") as f:
	for i in root_test_file.getchildren():
	    article = ' '.join(e for e in i.itertext()).lower()
	    v1 = model.infer_vector(article.lower())
	    similar_doc = model.docvecs.most_similar([v1])
	    if (similar_doc[0][1] > similar_doc[1][1]):
	        f.write(i.attrib['id'] + ' ' + similar_doc[0][0] + '\n')
	    else:
	        f.write(i.attrib['id'] + ' ' + similar_doc[1][0] + '\n')