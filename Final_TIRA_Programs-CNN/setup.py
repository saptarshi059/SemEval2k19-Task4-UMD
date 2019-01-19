from setuptools import setup

setup(
   name='unigram_CNN',
   version='1.0',
   description='The programs for running the CNN model approach',
   author='Saptarshi Sengupta',
   author_email='sengu059@d.umn.edu',
   packages=['unigram_CNN'],
   install_requires=['argparse', 'joblib', 'keras', 'lxml', 'tensorflow', 'tqdm'],
)