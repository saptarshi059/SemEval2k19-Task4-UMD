'''
Code Usage Instructions:

Run the program using the following command line instructions:

python ngram_train_v3.py --train <Path to training file> --train_label <Path to training labels> --ngramrange <Range of n-grams wanted as features> 
--cutoff <Select those terms which have a term frequency greater than this number> 
--onehot <Whether or not you want 1 hot encoding (default) or regular count based vectors> 
--op1 <Name to save the training model as> --op2 <Name of training vectors matrix> --op3 <Name to save the training labels as> 

Example Usage:

python ngram_train_v3.py --train /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/articles-training-bypublisher-20181122.xml --trainlabel /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/ground-truth-training-bypublisher-20181122.xml --ngrange 1 1 --cutoff 10 -op1 ngram_model -op2 trainvects -op3 model_labels 
python ngram_train_v3.py --train /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml --trainlabel /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 10 -op1 ngram_model -op2 trainvects -op3 model_labels 

python ngram_train_v3.py --train ~/Desktop/semeval2k19/data/train_data/train_bypub/articles-training-bypublisher-20181122.xml --trainlabel ~/Desktop/semeval2k19/data/train_data/train_bypub/ground-truth-training-bypublisher-20181122.xml --ngrange 1 1 --cutoff 10 -op1 ngram_model -op2 trainvects -op3 model_labels

cd ~/Desktop/semeval2k19/programs/sklearnprogs/SemEval2k19-Task4-UMD/Optimized_Ngram
'''

from sklearn.feature_extraction.text import HashingVectorizer
from collections import Counter
from nltk.util import ngrams #For getting bi & tri grams.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize #For getting unigrams.
from lxml import etree
from lxml import objectify
from tqdm import tqdm
from joblib import dump
import nltk
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-t','--train', metavar='', type=str, help='Path to training file (XML).' , required = True)
parser.add_argument('-tl','--trainlabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)
parser.add_argument('-ngr', '--ngrange' ,metavar='', nargs=2 ,type=int, help='Types of ngrams wanted as features: ex. for unigrams enter 1 1, unigrams and bigrams enter 1 2 etc.', required = True)
parser.add_argument('-c','--cutoff', metavar='', type=int, help='Select only those features which have frequency higher than this value.', required = True)
parser.add_argument('-oh','--onehot' , action='store_true' ,help='Whether or not you want the vectors to be one hot encoded. If yes, set/include this argument in the command line argument list else leave it.')
parser.add_argument('-op1', '--output1' ,metavar='', type=str, help='Name of the saved TDM model.', required = True)
parser.add_argument('-op2', '--output2' ,metavar='', type=str, help='Name of the training vectors matrix.', required = True)
parser.add_argument('-op3', '--output3' ,metavar='', type=str, help='Name of the labels matrix for the training file.', required = True)

args = parser.parse_args()

#Creating the xml object/tree
train_file = objectify.parse(open(args.train))
train_label_file = objectify.parse(open(args.trainlabel))

#To access the root element
root_train_file = train_file.getroot()
root_train_label_file = train_label_file.getroot()

training_corpus = []
training_labels = np.array([],dtype=object)

print "Reading in the training corpus:"
for i in tqdm(root_train_file.getchildren()):
	training_corpus.append(' '.join(e for e in i.itertext()))

'''
print "Reading in the training label file:"
for row in tqdm(root_train_label_file.getchildren()):
	training_labels = np.append(training_labels, row.attrib['hyperpartisan'])
'''

stop_words = set(stopwords.words('english'))

vectorizer = HashingVectorizer(ngram_range = args.ngrange , stop_words=stop_words, binary = args.onehot)
vectors = vectorizer.fit_transform(training_corpus)

Trained_Vectors = vectors.toarray()

sums = Trained_Vectors.sum(axis=0)

'''
print "Cleaning up the TDM according to the supplied cutoff value:"
j = 0
for i in tqdm(vocab.items()):
	if sums[i[1]] < args.cutoff:
		Trained_Vectors[:,i[1]] = -1
		del vocab[i[0]]
	j = j + 1
'''

print "The TDM Model, Training Vectors and Training Label matrix were saved..."
dump(vectorizer, args.output1+'.joblib')
np.save(args.output2, Trained_Vectors)
np.save(args.output3, training_labels)