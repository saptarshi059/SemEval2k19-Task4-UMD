# SemEval2k19-Task4-UMD

**Code and Project Advisor: Dr. Ted Pedersen</br>
Code Author: Saptarshi Sengupta</br>
Dept. of Computer Science, University of Minnesota Duluth**

### Introduction and Task Description ###

During SemEval 2019, the *Hyperpartisan News Detection* task was hosted. The objective of this task was to determine whether a given news article could be considered as hyperpartisan or not. Hyperpartisan news is a kind of *fake news* which is extremely polarized and written with the intent of creating political divide among the masses. Typically such kinds of articles have roots in political issues and it makes sense for them to do so. This is owing to the fact that bias in political news is generally able to incite more commotion than any other kind of news which in turn helps in adding more fuel to the preexisting fire surrounding political parties and leaders. Thus, it is of the essence that such articles be swiftly detected before they find their way to consumers and thus start their action.

As far as the task was concerned, each team had to come up with a model to detect such articles in a given dataset, typically in a supervised manner owing to the fact that each article was annotated i.e. as hyperpartisan or not. The dataset was provided by the authors of the paper, *A Stylometric Inquiry into Hyperpartisan and Fake News* (http://aclweb.org/anthology/P18-1022), which was the first real attempt at tackling this issue. The winner of the task was determined on the basis of a team's performance on the *articles-training-byarticle-20181122.xml* dataset [Link provided below] while results on the *articles-training-bypublisher-20181122.xml* dataset were also considered (for the overall task's purpose).

Our team submitted 2 models viz. a logistic regression classifier trained on unigrams having a term frequency greater than 12 and a CNN trained on word embeddings.

### A note on the programs ###

All of the programs found in this repo have been written with the *articles-training-byarticle-20181122.xml* dataset in mind. While the code can be extended to include a wider range of datasets viz. *articles-training-bypublisher-20181122.xml*, we had 2 reasons for using the former,

1. The final test data used for the competition (for determining the winner) was similar to the *articles-training-byarticle-20181122.xml* dataset.
2. The *articles-training-bypublisher-20181122.xml* dataset became quite difficult to parse owing to its large size which in turn became a bottleneck for our programs.

Thus we chose to try our programs on the *byarticle* data only. **Future versions of the code will have the capability of handling larger sized training data**

As we didn't have access to the final test data (which would be released after all the submissions were made), we had to find a way to understand how well our approaches would work on it. Thus, we decided to run each program with a ***10-fold cross-validation*** technique. In this way, we got hints about the final performance of our methods and in turn it helped us select the two models which we would be submitting for the task.

### Motivation for each approach ###

**In order to establish a baseline for each approach, we trained a *majority classifier* on each fold of the training data. This gave us a reasonable metric to evaluate our models.**

1. **Ngram**

	The first obvious approach for the task was to see how common ML classifiers were able to perform when trained using the simplest of features for text classification i.e. ngrams. For initial experiments, we chose unigram and bigram features over a range of cutoff/term-frequency values viz. 5,10,12,15 and 20. It was observed that performance declined when cutoff hit the 20 mark but peaked near the 10 - 15 mark. 
	We tested a range of classifiers including *Decision Trees, SVM, KNN, Logistic Regression and Naive Bayes*. In general, each classifier performed better when trained only on unigrams as opposed to bigrams. Furthermore, for a cutoff value of 12, each of them showed optimal performance with *logistic regression* (LR) becoming the winner of the lot. In second place came *SVM* followed by the others. Thus, we decided to select the LR model for our final submission as it produced an average accuracy of 77% using the configurations described above.

	A quick comparison of SVM and LR showed that their performances were neck and neck. The features used by both were almost similar over each fold. The differentiating factor between the two thus became the weights assignined to those features. For the same feature, it was seen that the LR classifier assigned almost 2 to 3 times the weight as assigned by SVM. The only way to confirm whether such weight differences were working or not, was so see the classification accuracies and surely enough, LR was beating SVM in almost every fold. Thus, the natural choice for submission became the LR model.

2. **CNN**

	As a comparative study to our Ngram model, we decided it would be interesting to see how a CNN (Convolutional Neural Network) fairs in the task using the same features as used by the LR model i.e. unigrams. Although a strange choice (as CNN's are usually used for image analysis tasks), we were curious to see how this approach would work, as neural networks are often state-of-the-art when it comes to such applications. Each article in the training fold was converted to a *sequence* of unigrams (which is a way by which *keras* (the module used to implement our CNN) handles data). These sequences or unigrams were in turn converted to their respective *embeddings*. Finally, the vectors were supplied to the network in order to train it. 

	What was most interesting about this approach was the number of options we could play around with, such as number of layers, size of the embeddings etc. This gives a lot of room for experimentation in order to see what combination of parameters gives optimal results.

	It was seen that for certain folds, the CNN would perform menacingly well, reaching accuracies of upto 90%. However, as the accuracy of classification from the other folds were not up to that mark, average accuracy hovered around the 75% mark. Thus, we decided to use this model for our second submission as it was performing the better than the remaining ones.

	*CNN implementation code was obtained from: https://realpython.com/python-keras-text-classification/*

3. **Doc2Vec**

	We explored a method of using *Document Vectors* in a very rudimentary way. Using *Doc2Vec* we trained a document vector model on the given training data and performed vector *inference* for the test data. Comparing the predicted and truth labels for the test data, the accuracy of classification was computed.

	Approaching the task from this angle seemed to make sense as there was a reasonable amount of training data and we felt it would be interesting to exlore this direction of directly vectorizing the articles in a dense format (rather than using raw term-frequency counts). However, we had to scrap this approach because the accuracy results were not up to the mark. We felt that the size of the data was the issue because *docvec* models typically tend to require larger volumes of data to make reasonable predictions. We believed that the number of *epochs* or iterations for the training process wasn't the issue since 100 iterations typically is a high enough setiing.

4. **HateSpeech**

	Before working on Task 4, I (Saptarshi) began working on Task 5 of *SemEval 2019* i.e. *HatEval: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter*. However, owing to a change of interest, I along with my advisor shifted our attention to Task 4. During the development phase for this task, we conjectured whether a model could be built by taking inspiration from the ideas which transpired during our discussions for Task 4. Thus, we designed a simple model which was loosely based on the unigram approach. We vectorized each article in the training fold as the count of the number of *"hate"* words (obtained from a list) present in it. This gave rise to an Nx1 dimension training matrix (where N = number of training articles) since each article was essentially represented as a single dimension.

	This approach gave moderately good results with SVM producing about 70% accuracy. The result was very similar to what it (SVM) was doing during the unigram approach. We felt that there could thus be some feature overlap. However, this was not the case. What did become clear though, was that hate words could definitely lend some insight into computationally understanding such articles. This is because, this approach provided evidence to a certain degree that there might be some influence of hate speech in hyperpartisan news articles, which was our intuition from the beginning. Surely there remains work to be done on extending the method to see whether better accuracy scores can be achieved.
	
	In spite of this approach doing well, we decided to nix it in favor of the CNN model owing to its higher accuracy.

	*List of hate words obtained from: http://www.frontgatemedia.com/new/wp-content/uploads/2014/03/Terms-to-Block.csv*

5. **Sentiment Analysis**

	An idea which came to mind during the development phase was seeing whether sentiment analysis could yield interesting results. We considered the possibility that since hyperpartisan news is biased content, it could have a high subjectivity score. On the other hand, as mainstream news is neutral in nature, we figured that it should have a low subjectivity rating. Thus, it would be easy to distinguish the two on such a basis. We also considered weights for the positive, negative, neutral and compound (is an overal sentiment score for the entire article) sentiments.

	We expected our intuition to work but instead, it resulted in average results nowhere near to the 77% mark set by the unigram LR model. Thus we decided to discard this approach as well. We believed that the reason for this poor performance was two fold. First, the packages being used to generate the scores weren't very good. We wish to try out the *Google* sentiment API for future experiments. Second, there wasn't enough training data to generate good enough scores. In order to solve this problem, we would like to train our own sentiment model on the given data (instead of using NLTK's pretrained model) and see whether that leads to enhanced performance.

6. **Image Presence**

	The last approach was based on analyzing an article's non-textual characteristics. The thought behind this approach was that since hyperpartisan news is more polarized etc., it needs more "flair" to convey its message. Mainstream news on the other hand can make its point with conviction without the need for such cues because its content is inherently true. Thus, we decided to see whether the number of images an article had could be used as a distinguishing factor or not. Unfortunately, this idea did not pan out the way we imagined. Accuracy scores for some of the classifiers were not even greater than the majority classifier. This could mean the following; the data did not reflect the idea proposed or other kinds of such features are required to engineer a better feature vector. Either way, we still need to look for such cues because we feel that the basic idea still has some merit to it. We feel that the next step to take would be to see whether we can use counts of images instead of a binary representation to improve accuracy.

### Usage Instructions ###

In order to execute the programs, one needs to create their own virtual environment. The steps are as follows:

- virtualenv -p python3 [environment_name] \(for unix based systems) || virtualenv [environment_name] \(for windows)
- source \path\to\virtual\environment\bin\activate (or activate.csh depending on how your system works) \(for unix based systems) || \path\to\virtual\environment\Scripts\activate \(for windows)
- pip install -r requirements.txt
- Execute programs as usual and exit virtual environment with deactivate.

**Please make sure you are using Python 3.6.x in order for tensorflow to work!**

[Change Log.](CHANGELOG.md)

---

## License & Copyright

© Saptarshi Sengupta & Ted Pedersen, University of Minnesota Duluth

Licensed under the [GNU GENERAL PUBLIC LICENSE v3.](LICENSE)