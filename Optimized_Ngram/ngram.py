#python ngram.py --train /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/samp/samp_train.xml --ngram 1 --topn 10
#python ngram.py --train /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp/samp_train.xml --ngram 1 --topn 10 --onehot False

from sklearn.feature_extraction.text import CountVectorizer
from lxml import etree
from lxml import objectify
from nltk.corpus import stopwords
import sys
import nltk
import ast

#Common code for All ngram pathways
stop_words_list = set(stopwords.words('english'))

#Creating the xml object/tree
train_file = objectify.parse(open(sys.argv[2]))

#To access the root element
root_train_file = train_file.getroot()

training_corpus = []

for i in root_train_file.getchildren():
	training_corpus.append(' '.join(e for e in i.itertext()))

vectorizer = CountVectorizer(analyzer = 'word', stop_words = stop_words_list , ngram_range = (int(sys.argv[4]),int(sys.argv[4])) , 
							max_features = int(sys.argv[6]) , binary = ast.literal_eval(sys.argv[8]))

vectors = vectorizer.fit_transform(training_corpus)
print vectorizer.vocabulary_
print vectors.toarray()