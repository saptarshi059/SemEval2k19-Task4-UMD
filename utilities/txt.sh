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

#You just need to modify these 2 variables according to where the files are located on your system.

DATA_HOME=~/Desktop/SemEval2k19/data/train_byarticle/
CODE_HOME=~/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/

#Convert training & test data/labels and create their corresponding txt files.

TRAIN_DATA=$DATA_HOME/articles-training-byarticle-20181122.xml
TRAIN_LABEL=$DATA_HOME/ground-truth-training-byarticle-20181122.xml
Train_Data_Output_Path=$DATA_HOME
Train_Label_Output_Path=$DATA_HOME

TEST_DATA=$TRAIN_DATA
TEST_LABEL=$TRAIN_LABEL
Test_Data_Output_Path=$Train_Data_Output_Path
Test_Label_Output_Path=$Train_Label_Output_Path

echo "python3 xml_to_txt.py -xp $TRAIN_DATA -tp $Train_Output_Path -m 1 -tetr 1"
python3 $CODE_HOME/utilities/xml_to_txt.py -xp $TRAIN_DATA -tp $Train_Data_Output_Path -m 1 -tetr 1

echo "python3 xml_to_txt.py -xp $TRAIN_LABEL -tp $Train_Output_Path -m 2 -tetr 1"
python3 $CODE_HOME/utilities/xml_to_txt.py -xp $TRAIN_LABEL -tp $Train_Label_Output_Path -m 2 -tetr 1

echo "python3 xml_to_txt.py -xp $TEST_DATA -tp $Test_Data_Output_Path -m 1 -tetr 2"
python3 $CODE_HOME/utilities/xml_to_txt.py -xp $TEST_DATA -tp $Test_Data_Output_Path -m 1 -tetr 2

echo "python3 xml_to_txt.py -xp $TEST_LABEL -tp $Test_Label_Output_Path -m 2 -tetr 2"
python3 $CODE_HOME/utilities/xml_to_txt.py -xp $TEST_LABEL -tp $Test_Label_Output_Path -m 2 -tetr 2

#Train LR classifier.

TRAIN_DATA=$DATA_HOME/train.txt
TRAIN_LABEL=$DATA_HOME/train_labels.txt

echo "python3 ngram_LR_train.py -t $TRAIN_DATA -tl $TRAIN_LABEL"
python3 $CODE_HOME/Logistic\ Regression/ngram_LR_train.py -t $TRAIN_DATA -tl $TRAIN_LABEL

#Generate Predictions.

TEST_DATA=$DATA_HOME/test.txt
PREDICTIONS_DIRECTORY_PATH=$DATA_HOME/predictions/

echo "python3 ngram_LR_predict.py -tf $TEST_DATA -o $PREDICTIONS_PATH"
python3 $CODE_HOME/Logistic\ Regression/ngram_LR_predict.py -tf $TEST_DATA -o $PREDICTIONS_DIRECTORY_PATH

#Evaluate Predictions.

TEST_LABEL=$DATA_HOME/test_labels.txt
PREDICTIONS_FILE_PATH=$DATA_HOME/predictions/predictions.txt

echo "python3 txt_evaluator.py -gp $TEST_LABEL -pp $PREDICTIONS_PATH"
python3 $CODE_HOME/utilities/txt_evaluator.py -gp $TEST_LABEL -pp $PREDICTIONS_FILE_PATH