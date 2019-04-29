from setuptools import setup

setup(
   name='ngram_LR',
   version='1.0',
   description='The programs for running the logisitic regression model approach',
   author='Saptarshi Sengupta',
   author_email='sengu059@d.umn.edu',
   packages=['ngram_LR'],
   install_requires=['joblib', 'lxml', 'nltk', 'tqdm', 'sklearn', 'matplotlib==2.2.3'],
)