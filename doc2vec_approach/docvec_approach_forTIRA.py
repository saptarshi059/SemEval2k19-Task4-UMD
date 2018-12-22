'''
This program will do the following:
1. Load the doc2vec model inside the code i.e. no need to pass it from the command line.
2. Generate the predictions of the input test file passed from the command line according to the required format (random_baseline).
'''

#python docvec_approach_forTIRA.py -i /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml

from __future__ import division
from gensim.models.doc2vec import Doc2Vec
from lxml import etree
from lxml import objectify
from tqdm import tqdm #For printing progress bars.
import sys

test_file = objectify.parse(open(sys.argv[2]))

root_test_file = test_file.getroot()

#Loading the model.
model = Doc2Vec.load('d2v.model')

predictions = open('prediction.txt','w')

for i in tqdm(root_test_file.getchildren()):
    article = ' '.join(e for e in i.itertext()).lower()
    v1 = model.infer_vector(article.lower())
    similar_doc = model.docvecs.most_similar([v1])
    if (similar_doc[0][1] > similar_doc[1][1]):
        predictions.write(i.attrib['id'] + ' ' + similar_doc[0][0] + '\n')
    else:
        predictions.write(i.attrib['id'] + ' ' + similar_doc[1][0] + '\n')