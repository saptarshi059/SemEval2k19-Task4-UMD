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

echo "This is a complete run of the xml version of the LR program."

#Train LR classifier.

TRAIN_DATA=~/Desktop/SemEval2k19/data/train_byarticle/data/articles-training-byarticle-20181122.xml
TRAIN_LABEL=~/Desktop/SemEval2k19/data/train_byarticle/labels/ground-truth-training-byarticle-20181122.xml

CODEHOME=/Users/babun/Desktop/SemEval2k19/programs/SemEval2k19-Task4-UMD/

echo "python3 ngram_LR_train.py -t $TRAIN_DATA -tl $TRAIN_LABEL"
python3 $CODEHOME/Logistic\ Regression/ngram_LR_train.py -t $TRAIN_DATA -tl $TRAIN_LABEL

#Generate Predictions.

TEST_DATA=~/Desktop/SemEval2k19/data/train_byarticle/data/articles-training-byarticle-20181122.xml
PREDICTIONS_PATH=~/Desktop/SemEval2k19/data/custom/test_data/predictions/

echo "python3 ngram_LR_predict.py -tf $TEST_DATA -o $PREDICTIONS_PATH"
python3 $CODEHOME/Logistic\ Regression/ngram_LR_predict.py -tf $TEST_DATA -o $PREDICTIONS_PATH

#Evaulate Predictions.

TEST_LABEL_DIRECTORY=~/Desktop/SemEval2k19/data/train_byarticle/labels/

echo "python3 semeval-pan-2019-evaluator.py -d $TEST_LABEL_DIRECTORY -r $PREDICTIONS_PATH -o $PREDICTIONS_PATH"
python3 $CODEHOME/utilities/semeval-pan-2019-evaluator.py -d $TEST_LABEL_DIRECTORY -r $PREDICTIONS_PATH -o $PREDICTIONS_PATH