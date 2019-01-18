#python docvec_approach.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/articles-training-bypublisher-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/articles-validation-bypublisher-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/ground-truth-training-bypublisher-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/ground-truth-validation-bypublisher-20181122.xml
#python docvec_approach.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/samp_train.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_bypub/samp_train_label.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test_label.xml

from __future__ import division
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from lxml import etree
from lxml import objectify
from tqdm import tqdm #For printing progress bars.
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

train_file = objectify.parse(open(sys.argv[1]))
test_file = objectify.parse(open(sys.argv[2]))

root_train_file = train_file.getroot()
root_test_file = test_file.getroot()

train_label_file = objectify.parse(open(sys.argv[3]))
test_label_file = objectify.parse(open(sys.argv[4]))

root_train_label_file = train_label_file.getroot()
root_test_label_file = test_label_file.getroot()

train_labels = []
test_labels = []

for i in tqdm(root_train_label_file.getchildren()):
    train_labels.append(i.attrib['hyperpartisan'])

for i in tqdm(root_test_label_file.getchildren()):
    test_labels.append(i.attrib['hyperpartisan'])

train_docs = []
test_docs = []

for i in tqdm(root_train_file.getchildren()):
    train_docs.append(' '.join(e for e in i.itertext()).lower())

for i in tqdm(root_test_file.getchildren()):
    test_docs.append(' '.join(e for e in i.itertext()).lower())

#Assigning hyperpartisan (true or false) tags to each document.
tagged_data = []
j = 0
for i in tqdm(train_docs):
    tagged_data.append(TaggedDocument(i.lower(),tags=[train_labels[j]]))
    j = j + 1

#Training The Doc2Vec Model. Changing epoch number might give better results but requires much more time.
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(tagged_data, vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1, epochs = 50)
'''
v1 = model.infer_vector(test_docs[1].lower())
similar_doc = model.docvecs.most_similar([v1])

print similar_doc

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
'''
#Saving the Model.
model.save("d2v.model")
print("Model Saved")

predictions = []

for i in tqdm(test_docs):
	v1 = model.infer_vector(i.lower())
	similar_doc = model.docvecs.most_similar([v1])
	if similar_doc[0][1] > similar_doc[1][1]:
		predictions.append(similar_doc[0][0])
	else:
		predictions.append(similar_doc[1][0])

correctly_classified = 0
j = 0

for i in tqdm(predictions):
	if i == test_labels[j]:
		correctly_classified = correctly_classified + 1
	j = j + 1

print "accuracy =",(correctly_classified/len(predictions)) * 100,"%"