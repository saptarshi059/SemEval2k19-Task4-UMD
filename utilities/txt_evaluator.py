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

from lxml import objectify
from joblib import load
import argparse

def save_to_file(filename):
	with open(filename, "w") as f:
		f.write("Accuracy = "+str(round(accuracy,2))+"%"+"\n")
		f.write("True Positives = "+str(tp)+"\n")
		f.write("False Negatives = "+str(fn)+"\n")
		f.write("False Positives = "+str(fp)+"\n")
		f.write("True Negatives = "+str(tn)+"\n")
		f.write("Precision = "+str(round(precision,2))+"\n")
		f.write("Recall = "+str(round(recall,2))+"\n")
		f.write("F1 = "+str(round(f1,2))+"\n")

parser = argparse.ArgumentParser(description="Prediction Evaluator for .txt formatted files.")

parser.add_argument('-gp','--groundtruthpath', metavar='', type=str, help='Path to the Ground Truth File.' , required = True)
parser.add_argument('-pp','--predictionspath', metavar='', type=str, help='Path to the Predictions File.', required = True)

args = parser.parse_args()

g = open(args.groundtruthpath,"r").readlines()
p = open(args.predictionspath,"r").readlines()

correct = 0

tp = tn = fp = fn = 0

for i in range(len(g)):
	prediction = p[i].split()[1]
	ground_truth = g[i].replace('\n','')

	#Accuracy
	if prediction == ground_truth:
		correct += 1

	#Confusion Matrix
	if ground_truth == 'true':
		if prediction == 'true':
			tp += 1
		else:
			fn += 1
	else:
		if prediction == 'false':
			tn += 1
		else:
			fp += 1

accuracy = (correct/len(g)) * 100

print("Accuracy = "+str(round(accuracy,2))+"%")
print("True Positives =",tp)
print("False Negatives =",fn)
print("False Positives =",fp)
print("True Negatives =",tn)

precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)

print("Precision =",round(precision,2))
print("Recall =",round(recall,2))
print("F1 =",round(f1,2))

print("<Do you wish to save these readings(Y/N)?")
choice1 = input(">")
if choice1.upper() == "Y":
	print("<Do you want to enter a filename(Y/N)?")
	choice2 = input(">")
	if choice2.upper() == "Y":
		print("<Enter File Name")
		filename = input(">")
		save_to_file(filename)
	else:
		save_to_file("readings.txt")
else:
	exit()