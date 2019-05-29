# Change Log

## v1.0.11 - May 29 2019

2 shell scripts have been added to the utilities folder which demonstrates how to run the programs completely from conversion to txt to evaluation. Before running the scripts, make sure you enter **chmod 700 [name of the script]** in the terminal. Also, change the files paths according to where the data is located on your machine.

## v1.0.10 - May 28 2019

**Bug Fix**
- An issue with the reading of the training labels in the training programs and subsequent use in the predictor programs has been fixed.

## v1.0.9 - May 27 2019

Comments in each file have been updated to reflect the inclusion of ".txt" files as input.
Programs in *all_approaches & results* still use ".xml" as input only. Future work entails development on those programs.

## v1.0.8 - May 26 2019

**Bug Fix**
- An issue with the reading of the training labels in the training programs has been fixed.

**Added**
- All programs have been updated with license information.

---

## v1.0.7 - May 24 2019

2 new programs have been added in the *utilities* folder for working with ".txt" files. "xml_to_txt.py" converts your supplied xml file to the required ".txt" equivalent for use with our programs. "txt_evaluator.py" is used to evaluate the predicitions generated from the ".txt" files. In order to evaluate predictions generated using ".xml" files, use the semeval evaluator script located here https://pan.webis.de/semeval19/semeval19-code/semeval-pan-2019-evaluator.py.

In order to download the official data, you need to request access at https://zenodo.org/record/1489920#.XOisOdNKi9Y.

---

## v1.0.6 - May 17 2019

**Bug Fix**
- Provided virtual environments have been removed due to portability issues.

In order to execute the programs, one needs to create their own virtual environment. The steps are as follows:

- virtualenv -p python3 [environment_name] \(for unix based systems) || virtualenv [environment_name] \(for windows)
- source \path\to\virtual\environment\bin\activate (or activate.csh depending on how your system works) \(for unix based systems) || \path\to\virtual\environment\Scripts\activate \(for windows)
- pip install -r requirements.txt
- Execute programs as usual and exit virtual environment with deactivate.

**Please make sure you are using Python 3.6.x in order for tensorflow to work!**

---

## v1.0.5 - May 15 2019

**Main programs have been updated to work with Python 3**. Setup scripts have been removed and have been replaced with virtual environments. Please run the programs in *Logistic Regression* and *CNN* by using their respective virtual environments. In order to use the programs now, enter the following commands:

- source \path\to\virtual\environment\bin\activate
- pip install -r requirements.txt
- run programs as usual.

Once you are done with the enironment, enter *deactivate* in the prompt to exit.

There is no need to set the *COLUMNS* environment variable now as that has been taken care of within the code.

License (GPLv3) to the code has been added.

---

## v1.0.4 - April 27 2019

Support for using text files have been added. They are to be supplied to the program in the same manner as .xml files. Ability to see features and their corresponding weights from the Logistic Regression approach has been added. However, in order to use the LR program now, use *COLUMNS=81* followed by the rest of the command line arguments. This is required now because of the way *argparse* works when the number of command line options are many.

---

## v1.0.3 - 3rd February 2019

Programs in the *all_approaches & results* folder have been updated with comments.

---

## v1.0.2 - 19th January 2019

Each final approach (previously located in the *Final_TIRA_Programs* folder), now has its own directory. Located within each directory is a *setup.py* program which will install all the dependencies required to run that particular approach. In order to use it, run *python setup.py install*. Thus, the only prerequisite now is having the python compiler installed.

---

## v1.0.1 - 17th January 2019

The repository has been cleaned and the only folder which is of importance as far as SemEval is concerned is the *Final_TIRA_Programs*. These programs were designed for use with the *TIRA* system. This folder holds the programs for the two approaches we selected for the competition. One of the them is a logistic regression classifier trained on a unigram model. Although the program is called *ngram_LR_train* etc., we use only unigrams as features. The name was so given because the program is capable of handling n(>1)gram features. The other is a CNN trained on a unigram/embedding model obtained from the supplied training data.

The folder named *all_approaches* contains programs for all the approaches we tried for the problem. All of them are fully functional. However, they are not formatted according to the SemEval standards. Running them is fairly simple if one looks at the help menu obtained from the command line using the '-h' flag. These programs were mostly based on a kfold cross-validation technique and were written with the *articles-training-byarticle-20181122.xml* dataset (https://zenodo.org/record/1489920) in mind.

The **ENTIRE** list of dependencies which are required to run *all* the programs are as follows:
- **Python**
- **NumPy**
- **scikit-learn**
- **SciPy**
- **Tensorflow**
- **Keras**
- **tqdm**
- **lxml**
- **argpars**
- Gensim
- Six
- smart_open
- NLTK
- TextBlob
- CSV
- re
- Matplotlib (optional)

In order to run the programs in *Final_TIRA_Programs*, one needs to only have the **first 9 dependencies** installed.

---

## v1.0.0 - 29th November 2018

Initial Release.

There are 3 files, the 'generate_vectors.py' and 'generate_sentiment_vectors.py' are used to create unigram/image presence and sentiment vectors respectively. The 'classify.py' is used for training and testing the SVM, Gaussian NB and Decision Tree classifiers.

There are no output files for the last one as the results are displayed directly in the terminal. For the first 2, the results are stored as '.csv' files. In order to run the classification program, the 2 output files from the first 2 programs are manually merged into files called 'train.csv' and 'test.csv'.

The first few lines of each program are commented out. They are simply the command line arguments I used to run the program. You can change it accordingly to where you have kept your files.

**All Programs were written using Python 2.7. Please run them accordingly!**

The repository is organized in the following manner.
1. The *Final_TIRA_Programs-CNN* directory contains the programs for our CNN model.
2. The *Final_TIRA_Programs-Logistic Regression* directory contains the programs for our Logistic Regression model.
3. The *all_approaches & results* directory contains code for each approach that was attempted for the task.

Each program in the first 2 directories is documented/commented such that it explains what the program is doing and provides instructions on running it. **Code in the 3rd directory will be updated with comments shortly but the process of running them is almost the same as those in the final directories.**