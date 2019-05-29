#!/bin/sh
: '
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
Email: sengu059@d.umn.edu
'

echo "This is a complete run of the txt version of the LR program."

#convert training, test data/labels and create their corresponding txt files.

TRAIN_DATA=~/Desktop/SemEval2k19/data/custom/train_data/train.xml
TRAIN_LABEL=~/Desktop/SemEval2k19/data/custom/train_data/train_labels.xml
Train_Output_Path=~/Desktop/SemEval2k19/data/custom/train_data/

TEST_DATA=~/Desktop/SemEval2k19/data/custom/test_data/data/test.xml
TEST_LABEL=~/Desktop/SemEval2k19/data/custom/test_data/ground_truth/test_labels.xml
Test_Data_Output_Path=~/Desktop/SemEval2k19/data/custom/test_data/data/
Test_Label_Output_Path=~/Desktop/SemEval2k19/data/custom/test_data/ground_truth/

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TRAIN_DATA -tp $Train_Output_Path -m 1 -tetr 1"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TRAIN_DATA -tp $Train_Output_Path -m 1 -tetr 1

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TRAIN_LABEL -tp $Train_Output_Path -m 2 -tetr 1"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TRAIN_LABEL -tp $Train_Output_Path -m 2 -tetr 1

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TEST_DATA -tp $Test_Data_Output_Path -m 1 -tetr 2"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TEST_DATA -tp $Test_Data_Output_Path -m 1 -tetr 2

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TEST_LABEL -tp $Test_Label_Output_Path -m 2 -tetr 2"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/xml_to_txt.py -xp $TEST_LABEL -tp $Test_Label_Output_Path -m 2 -tetr 2

#Train LR classifier.

TRAIN_DATA=~/Desktop/SemEval2k19/data/custom/train_data/train.txt
TRAIN_LABEL=~/Desktop/SemEval2k19/data/custom/train_data/train_labels.txt

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/Logistic\ Regression/ngram_LR_train.py -t $TRAIN_DATA -tl $TRAIN_LABEL"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/Logistic\ Regression/ngram_LR_train.py -t $TRAIN_DATA -tl $TRAIN_LABEL

#Generate Predictions.

TEST_DATA=~/Desktop/SemEval2k19/data/custom/test_data/data/test.txt
PREDICTIONS_PATH=~/Desktop/SemEval2k19/data/custom/test_data/predictions/

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/Logistic\ Regression/ngram_LR_predict.py -tf $TEST_DATA -o $PREDICTIONS_PATH"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/Logistic\ Regression/ngram_LR_predict.py -tf $TEST_DATA -o $PREDICTIONS_PATH

#Evaulate Predictions.

TEST_LABEL=~/Desktop/SemEval2k19/data/custom/test_data/ground_truth/test_labels.txt
PREDICTIONS_PATH=~/Desktop/SemEval2k19/data/custom/test_data/predictions/predictions.txt

echo "/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/txt_evaluator.py -gp $TEST_LABEL -pp $PREDICTIONS_PATH"
python3 /Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/utilities/txt_evaluator.py -gp $TEST_LABEL -pp $PREDICTIONS_PATH