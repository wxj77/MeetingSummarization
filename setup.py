from setuptools import setup, find_packages

setup(
   name='MeetingSummarization',
   version='1.0',
   description='A meeting summarization module',
   author='Andreas Huebner, Wei Ji, Xiang Xiao',
   author_email='jiwei0706@gmail.com, mitchell.xiao@gmail.com, andreas.huebnerh@gmail.com',
   packages=find_packages(),  #same as name
   install_requires=['wheel',], #external packages as dependencies
)
