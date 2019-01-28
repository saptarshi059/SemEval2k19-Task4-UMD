# SemEval2k19-Task4-UMD

**Introduction and Task Description**

During SemEval 2019, the *Hyperpartisan News Detection* task was hosted. The objective of this task was to determine whether a given news article could be considered as hyperpartisan or not. Hyperpartisan news is kind of *fake news* which is extremely polarized and written with the intent of creating a political divide among readers. Typically such kinds of articles have roots in political issues and it makes sense for them to be so. This is owing to the fact that bias in political news is generally able to incite more commotion than any other kind of news which in turn helps in adding more fuel to the preexisting fire surrounding political parties and leaders. Thus, it is of the essence that such articles be swiftly detected before they find their way to consumers and thus start their destruction.

As far as the task was concerned, each team had to come up with a model to detect such articles in a given dataset, typically in a supervised manner owing to the fact that each article was annotated. The dataset provided was created by the authors of the paper *A Stylometric Inquiry into Hyperpartisan and Fake News* (http://aclweb.org/anthology/P18-1022).

**29th November 2018**

There are 3 files, the 'generate_vectors.py' and 'generate_sentiment_vectors.py' are used to create unigram/image presence and sentiment vectors respectively. The 'classify.py' is used for training and testing the SVM, Gaussian NB and Decision Tree classifiers.

There are no output files for the last one as the results are displayed directly in the terminal. For the first 2, the results are stored as '.csv' files. In order to run the classification program, the 2 output files from the first 2 programs are manually merged into files called 'train.csv' and 'test.csv'.

The first few lines of each program are commented out. They are simply the command line arguments I used to run the program. You can change it accordingly to where you have kept your files.

**Update 17th January 2019**

The repository has been cleaned and the only folder which is of importance as far as SemEval is concerned is the *Final_TIRA_Programs*. These programs were designed for use with the *TIRA* system. This folder holds the programs for the two approaches we selected for the competition. One of the them is a logistic regression classifier trained on a unigram model. Although the program is called *ngram_LR_train* etc., we use only unigrams as features. The name was so given because the program is capable of handling n(>1)gram features. The other is a CNN trained on a unigram/embedding model obtained from the supplied training data.

Each program in this folder is well documented so that a user may have as much as background information as possible regarding the approach/code.

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

**Update 19th January 2019**

Each final approach (previously located in the *Final_TIRA_Programs* folder), now has its own directory. Located within each directory is a *setup.py* program which will install all the dependencies required to run that particular approach. In order to use it, run *python setup.py install*. Thus, the only prerequisite now is having the python compiler installed.

*Code/Project Advisor: Dr. Ted Pedersen*
