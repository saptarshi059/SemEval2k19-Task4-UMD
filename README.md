# SemEval2k19-Task4-UMD

There are 3 files, the 'generate_vectors.py' and 'generate_sentiment_vectors.py' are used to create unigram/image presence and sentiment vectors respectively. The 'classify.py' is used for training and testing the SVM, Gaussian NB and Decision Tree classifiers.

There are no output files for the last one as the results are displayed directly in the terminal. For the first 2, the results are stored as '.csv' files. In order to run the classification program, the 2 output files from the first 2 programs are manually merged into files called 'train.csv' and 'test.csv'.

The first few lines of each program are commented out. They are simply the command line arguments I used to run the program. You can change it accordingly to where you have kept your files.

**Update 17th January 2019**

The repository has been cleaned and the only folder which is of importance as far as SemEval is concerned is the *Final_TIRA_Programs*. These programs were designed for use with the *TIRA* system. This folder holds the programs for the two approaches we selected for the competition. 
	One of the them is a logistic regression classifier trained on a unigram model. Although the program is called *ngram_LR_train* etc., we use only unigrams as features. The name was so given because the program is capable of handling n(>1)gram features.
	The other is a CNN trained on a unigram/embedding model.

Each program in this folder is well documented so that a user may have as much as background information as possible regarding the approach/code.

The folder named *all_approaches* contains programs for all the approaches we tried for the problem. All of them are fully functional. However, they are not formatted according to the SemEval standards. Running them is fairly simple if one looks at the help menu obtained from the command line using the '-h' flag. These programs were mostly based on a kfold cross-validation technique and were written with the *articles-training-byarticle-20181122.xml* dataset in mind.

The **ENTIRE** list of dependencies which are required to run *all* the programs are as follows:
- Python
- NumPy
- scikit-learn
- SciPy
- Tensorflow
- Keras
- tqdm
- lxml
- argpars
- Gensim
- Six
- smart_open
- NLTK
- TextBlob
- CSV
- re
- Matplotlib (optional)

In order to run the programs in *Final_TIRA_Programs*, one needs to have the **first 9 dependencies** installed.

*Code/Project Advisor: Dr. Ted Pedersen*