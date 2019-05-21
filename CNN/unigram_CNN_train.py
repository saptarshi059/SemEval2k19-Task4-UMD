'''
Program Name: unigram_CNN_train.py
Author: SAPTARSHI SENGUPTA
Major: Computer Science / 1st Year / Graduate Student (MS) / University of Minnesota Duluth

Program Details: The unigram_CNN_train program trains a Convolutional Neural Network (CNN) on a unigram Language Model (LM) learned from a set of training articles. Each unigram is converted to an embedding which gives rise to a dense representation of each article's vector. We set the following specifications for our model/system:

				* We set a maximum vector length of 10000. This allows all the vectors to have a fixed size and it is achieved through padding. With padding, those vectors which are less than the specified size are padded with 0's (we use post padding) and those which are longer than 10000, are truncated down.

				* We use 100 epochs to train our model.

				* The length of each embedding was set to 100.

				* Our model has a 'GlobalMaxPooling1D' layer succeding the embedding layer. This allows the model to reduce the size of the input feature vectors for further denser representations. 

				* The optimizing algorithm for the loss function was 'Adam'.

Finally, the training model and the tokenizer are persisted to the disk for use with the 'predictor' program.

Code Usage: In order to use this program -
				* A user must have the Python programming language compiler installed and MUST have keras and Tensorflow installed.
				
				* The user should type the following command in either command prompt or linux shell i.e. 
				  				python unigram_CNN_train.py -t <path to training data file> -tl <path to training data's label file> -mn <Name of the saved CNN model> -tn <Name of the saved tokenizer>

			    In the above prompt, 

				* The training file and its corresponding labels file MUST be in XML format for the program to work.
				
				* An example usage would be of the form: python unigram_CNN_train.py --train articles-training-byarticle-20181122.xml --trainlabel ground-truth-training-byarticle-20181122.xml -mn myCNN -tn mytokenizer

Program Algorithm: The program has the following underlying logic -
				 
				//PreProcessing

				1. Each article was tokenized into a vector of unigrams.
				2. Each vector was then converted to a sequence of numbers where each number is an index of a word in the vocabulary. This was done using a built in function of keras. Also, this was done because of the way keras handles neural networks.
				3. The vectors were then padded with 0's in order to reach a length of 10000. Vectors which were greater in length were skimmed down to 10000.
				4. Finally, all the sequence numbers get replaced by their embedding vector of length 100.
				
				//Main Program Logic

				1. At first, the XML training file is parsed so as to retrieve each article which in turn is stored in memory as a python list. The same takes place for the training labels file.

				2. All the preprocessing steps are carried out.

				3. The model is trained on the training vectors over 100 epochs.

				4. The training model and the tokenizer are saved.

Future Work: The number of directions we could go from here are quite a few owing to a neural networks excitingly programmable nature. Varying the parameters of the model such as the maximum size of the training vectors, embeddings, number of epochs, is the first thing which comes to mind. Changing the activation functions, optimizer or even adding more layers, could have an effect on the model performance. Finally, as this model only used word embeddings as features, it would be interesting to see the effect other features might have on the model in conjunction with the embeddings.
'''

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.models import load_model
from lxml import objectify
from keras import layers
from joblib import dump
from lxml import etree
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='Train and save a CNN model on the supplied Training data')

parser.add_argument('-t','--train', metavar='', type=str, help='Path to training file (XML).' , required = True)
parser.add_argument('-tl','--trainlabel', metavar='', type=str, help='Path to training files labels (XML).', required = True)
parser.add_argument('-mn','--modelname', metavar='', type=str, help='Name of the saved CNN model', default='myCNN')
parser.add_argument('-tn','--tokenizername', metavar='', type=str, help='Name of the saved tokenizer', default='mytokenizer')

args = parser.parse_args()

#Check to see whether the file exists or not.
exists_train_file = os.path.isfile(args.train)
exists_train_label_file = os.path.isfile(args.trainlabel)

if exists_train_file and exists_train_label_file:
	#Checking File extension
	if args.train.endswith('.xml') and args.trainlabel.endswith('.xml'):
		#Creating the xml object/tree
		training_file = objectify.parse(open(args.train))
		training_label_file = objectify.parse(open(args.trainlabel))

		#To access the root element
		root_data_file = training_file.getroot()
		root_data_label_file = training_label_file.getroot()

		training_data = []
		training_labels = []

		print("Reading in the training corpus:")
		for i in tqdm(root_data_file.getchildren()):
			training_data.append(' '.join(e for e in i.itertext()))

		print("Reading in the training label file:")
		for row in tqdm(root_data_label_file.getchildren()):
			if row.attrib['hyperpartisan'] == 'true':
				training_labels.append(1)
			else:
				training_labels.append(0)

	elif args.train.endswith('.txt') and args.trainlabel.endswith('.txt'):
		print("Reading in the training corpus:")
		training_data = open(args.train,'r').readlines()

		print("Reading in the training label file:")
		training_labels = open(args.trainlabel,'r').readlines()

	else:
		print("Provided files extensions do not match. Program is now exiting...")
		exit()

elif exists_train_file == False:
	print("The training file does not exist! Program is now exiting...\n")
	exit()

else:
	print("The test file does not exist! Program is now exiting...\n")
	exit()

maxlen = 10000

tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(training_data)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
train_vectors = tokenizer.texts_to_sequences(training_data)
train_vectors = pad_sequences(train_vectors, padding='post', maxlen=maxlen)

embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_vectors, training_labels, epochs=100, verbose=1, batch_size=10)

#Saving the model
model.save(args.modelname)

#Saving the tokenizer
dump(tokenizer, args.tokenizername)