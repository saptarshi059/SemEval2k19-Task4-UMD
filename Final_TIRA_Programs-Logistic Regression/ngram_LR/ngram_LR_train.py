'''
Program Name: ngram_LR_train
Author: SAPTARSHI SENGUPTA
Major: Computer Science / 1st Year / Graduate Student (MS) / University of Minnesota Duluth

Program Details: The ngram_LR_train program has been written for training a Logistic Regression (LR) classifier on an Ngram Language Model (LM) learned from a set of training articles. The program generates a term-document matrix using python's 'scikit-learn countvectorizer'. This allows us to convert each article into a vector of word counts, where each word is a feature and is unique to the vocabulary of the entire corpus. Finally, the LR classifier is trained on this model and it is persisted to the disk along with the TDM itself for use in the 'predictor' program.

Code Usage: In order to use this program -
				* A user must have the Python programming language compiler installed.
				
				* The user should type the following command in either command prompt or linux shell i.e. 
				  				python ngram_LR_train.py -t <path to training data file> -tl <path to training data XML file> -ngr [m,n] -c <cutoff value> -oh (optional) -tdmn <name of the saved TDM model> -lrmn <name of the saved LR model>

			    In the above prompt, 

				* The training file and its corresponding labels file MUST be in XML format for the program to work.

				* 'ngr' is used to specify the range of ngram features wanted. If only one kind of ngram features are desired, for ex, only unigrams, then m = n i.e. '1 1' should be provided. If we want a combination of unigrams and bigrams, [m,n] = [1,2] should be specified etc.

				* 'cutoff' is used to select those features which have a term frequency HIGHER than this value.

				* 'oh' specifies whether we want our vectors to be 'one hot encoded' or not. If yes, include this parameter else do not.
				
				* An example usage would be of the form: python ngram_LR_train.py -t /Users/babun/Desktop/SemEval2k19/data/train_byarticle/articles-training-byarticle-20181122.xml -tl /Users/babun/Desktop/SemEval2k19/data/train_byarticle/ground-truth-training-byarticle-20181122.xml --ngrange 1 1 --cutoff 12 -tdmn TDM -lrmn LR

Settings Used:  * Unigram features.
				* Term frequence cutoff 12.

Program Algorithm: The program has the following underlying logic -
				 
				//PreProcessing - Applies to all articles
	
				1. Each word is converted to lowercase. 
				2. Stop words are ignored.
				3. Finally, every non-word character and digits are ignored.

					It has been mentioned 'ignored' and not removed because of the way 'CountVectorizer' works. It allows one to specify a list of stopwords to ignore and what qualifies as a token using a regular expression. 

				//Main Program Logic

				1. At first, the XML training file is parsed so as to retrieve each article which in turn is stored in memory as a python list. The same takes place for the training labels file.

				2. The vectorizer is trained on all of these articles and generates a set of count vectors. This is the TDM, where each row indicates a document and each column, a term from the vocabulary (the vectorizer computes a vocabulary of terms automatically for the entire corpus). Each cell of the TDM contains the frequency (count) of a term in a given document.
				
					These vector are then subjected to further processing.

				3. By computing the sum of each column of the TDM, the frequency of each term (over the entire corpus) is obtained. If the frequency for a given term is less than the specified cutoff value, it is dropped from the vocabulary. In this way, the initial vocabulary gets updated.

				4. Next, the vectorizer is reinstantiated with the updated vocabulary and trained on the training articles to generate the final TDM containing the final training vectors.

				5. The LR classifier is then trained with these vectors and their corresponding truth labels.

		    	6. Finally, the TDM and the LR are persisted (saved) to the disk for use with the predictor program.

Future Work: There are a few pathways we can follow post this experiment. Firstly, it would be interesting to see whether 
			 using a tfidf matrix could give improvements in terms of classification accuracy. Although we filter out stop words, the effect of using a metric which allows an encapsulation of a term's importance in a corpus would surely be interesting to see when training the LR classifier.
			 The LR classifier associates weights with the different classes in the training dataset. As we only used the classifier in the default settings, each class (hyperpartisan/mainstream) was set to have the same weight (one). It would be interesting to see whether incorporating such information would have an effect on the classifiers performance.
			 Finally, as the size of the training and testing data is relatively small, we would like to explore a very rudimentary approach called decision lists which borrows some ideas from the unigram model proposed but would serve as a good baseline comparison model. <I could delete this point if it doesn't fit with the overall text.>
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from lxml import objectify
from joblib import dump
from lxml import etree
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Create TDM matrix and Training Vectors for the supplied Training file')

parser.add_argument('-t','--train', metavar='', type=str, help='Path to training file (XML).' , required = True)
parser.add_argument('-tl','--trainlabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)
parser.add_argument('-ngr', '--ngrange' ,metavar='', nargs=2 ,type=int, help='Types of ngrams wanted as features: ex. for unigrams enter 1 1, unigrams and bigrams enter 1 2 etc.', required = True)
parser.add_argument('-c','--cutoff', metavar='', type=int, help='Select only those features which have frequency higher than this value.', required = True)
parser.add_argument('-oh','--onehot' , action='store_true' ,help='Whether or not you want the vectors to be one hot encoded. If yes, set/include this argument in the command line argument list else leave it.')
parser.add_argument('-tdmn','--tdmname', help='Name of the saved TDM model', required=True)
parser.add_argument('-lrmn','--lrmname', help='Name of the saved LR model', required=True)

args = parser.parse_args()

#Creating the xml object/tree
training_file = objectify.parse(open(args.train))
training_label_file = objectify.parse(open(args.trainlabel))

#To access the root element
root_data_file = training_file.getroot()
root_data_label_file = training_label_file.getroot()

training_data = []
training_labels = []

print "Reading in the training corpus:"
for i in tqdm(root_data_file.getchildren()):
	training_data.append(' '.join(e for e in i.itertext()))

print "Reading in the training label file:"
for row in tqdm(root_data_label_file.getchildren()):
	training_labels.append(row.attrib['hyperpartisan'])

stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer(ngram_range = args.ngrange , stop_words=stop_words, binary = args.onehot, analyzer = 'word', token_pattern = r'\b[^\W\d]+\b' )
vectors = vectorizer.fit_transform(training_data).toarray()

sums = vectors.sum(axis=0)
j = 0
print "Cleaning up the TDM according to the supplied cutoff value:"
for i in tqdm(vectorizer.vocabulary_.items()):
	if sums[i[1]] < args.cutoff:
		del vectorizer.vocabulary_[i[0]]
	j = j + 1

#We have to do this step because the test vectors will have to be transformed on a new TDM i.e. the one which reflects the dropped features.
j = 0
for key in vectorizer.vocabulary_.keys():
	vectorizer.vocabulary_[key] = j
	j = j + 1

#Supplying the new vocabulary here.
print "Retraining the vectorizer with the new vocabulary:"
vectorizer = CountVectorizer(ngram_range = args.ngrange , stop_words=stop_words , vocabulary=vectorizer.vocabulary_ , binary = args.onehot, analyzer = 'word', token_pattern = r'\b[^\W\d]+\b')
training_vectors = vectorizer.fit_transform(training_data).toarray()

print "Features Used are:"
print vectorizer.vocabulary_.keys()
print "\n"

#Classifier Object
lr = LogisticRegression()

#Training the Classifier
lr_clf = lr.fit(training_vectors, training_labels)

dump(vectorizer, args.tdmname + '.joblib')
dump(lr_clf, args.lrmname + '.joblib')
print "The TDM and LR model was saved..."