#python classify.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/ground-truth-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/samp_test_label.xml
#python classify.py /home/csgrads/sengu059/Desktop/semeval2k19/data/train_data/train_byarticle/ground-truth-training-byarticle-20181122.xml /home/csgrads/sengu059/Desktop/semeval2k19/data/test_data/ground-truth-validation-bypublisher-20181122.xml

#python classify.py /Users/babun/Desktop/SemEval2k19/data/train_bypublisher/samp_train_label.xml /Users/babun/Desktop/SemEval2k19/data/test/samp_test_label.xml

from __future__ import division
from lxml import etree
from lxml import objectify
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import sys
import csv

train_label_file = objectify.parse(open(sys.argv[1]))
test_label_file = objectify.parse(open(sys.argv[2]))

root_train_label_file = train_label_file.getroot()
root_test_label_file = test_label_file.getroot()

train_labels = []
test_labels = []

train_vectors_numerized = []
test_vectors_numerized = []

for i in root_train_label_file.getchildren():
	train_labels.append(i.attrib['hyperpartisan'])

for i in root_test_label_file.getchildren():
	test_labels.append(i.attrib['hyperpartisan'])

with open('train.csv', 'r') as f:
  reader = csv.reader(f)
  for v in reader:
  	train_vectors_numerized.append(map(float,list(v))) 

with open('test.csv', 'r') as f:
  reader = csv.reader(f)
  for v in reader:
	test_vectors_numerized.append(map(float,list(v)))


#Classifier Object
dt = tree.DecisionTreeClassifier()
svm = svm.SVC(gamma='auto',kernel='rbf')
gnb = GaussianNB()

#Training the Classifier
dt_clf = dt.fit(train_vectors_numerized,train_labels)
svm_clf = svm.fit(train_vectors_numerized,train_labels)
gnb_clf = gnb.fit(train_vectors_numerized,train_labels)

#Classification
dt_predictions = dt_clf.predict(test_vectors_numerized)
svm_predictions = svm_clf.predict(test_vectors_numerized)
gnb_predictions = gnb_clf.predict(test_vectors_numerized)

#Accuracy
correctly_classified = 0
j = 0
for i in dt_predictions:
	if i == test_labels[j]:
		correctly_classified = correctly_classified + 1
	j = j + 1

dt_accuracy = (correctly_classified / len(dt_predictions)) * 100

correctly_classified = 0
j = 0
for i in svm_predictions:
	if i == test_labels[j]:
		correctly_classified = correctly_classified + 1
	j = j + 1

svm_accuracy = (correctly_classified / len(svm_predictions)) * 100

correctly_classified = 0
j = 0
for i in gnb_predictions:
	if i == test_labels[j]:
		correctly_classified = correctly_classified + 1
	j = j + 1

gnb_accuracy = (correctly_classified / len(gnb_predictions)) * 100

print "Decision Tree accuracy =",dt_accuracy,"%"
print "SVM accuracy=",svm_accuracy,"%"
print "GNB accuracy=",gnb_accuracy,"%"