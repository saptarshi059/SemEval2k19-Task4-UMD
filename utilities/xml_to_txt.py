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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Utility program to convert supplied xml file to txt file for usage with the given programs.')

parser.add_argument('-xp','--xmlpath', metavar='', type=str, help='Path to the training file (.xml).' , required = True)
parser.add_argument('-tp','--txtpath', metavar='', type=str, help='Path to where the converted training file (.txt) will be saved.', required = True)
parser.add_argument('-m','--mode', metavar='', type=int, help='Mode 1 = Convert data file; Mode 2 = Convert label file.', choices=[1,2], required = True)
parser.add_argument('-tetr','--testortrain', metavar='', type=int, help='Mode 1 = Convert train file; Mode 2 = Convert test file.', choices=[1,2], required = True)

args = parser.parse_args()

if args.mode == 1:
	xml_data_file = objectify.parse(open(args.xmlpath,encoding="utf-8"))
	xml_data_file_root = xml_data_file.getroot()

	articles = []

	if args.testortrain == 1:
		
		for i in tqdm(xml_data_file_root.getchildren()):
			articles.append(' '.join(e for e in i.itertext()).replace('\n',' '))

		with open(args.txtpath + "/train.txt", 'w') as fp:
			fp.write('\n'.join('%s' % x for x in articles))

	else:
		
		for i in tqdm(xml_data_file_root.getchildren()):
			articles.append((i.attrib['id'],' '.join(e for e in i.itertext()).replace('\n',' ')))
		
		with open(args.txtpath + "/test.txt", 'w') as fp:
			fp.write('\n'.join('%s %s' % x for x in articles))

else:

	xml_train_label_file = objectify.parse(open(args.xmlpath,encoding="utf-8"))
	xml_train_label_file_root = xml_train_label_file.getroot()

	train_labels = []

	for i in tqdm(xml_train_label_file_root.getchildren()):
		train_labels.append(i.attrib['hyperpartisan'])

	if args.testortrain == 1:

		with open(args.txtpath + "/train_labels.txt", 'w') as fp:
			fp.write('\n'.join('%s' % x for x in train_labels))

	else:

		with open(args.txtpath + "/test_labels.txt", 'w') as fp:
			fp.write('\n'.join('%s' % x for x in train_labels))		