from setuptools import setup

setup(
   name='nsight',
   version='1.0',
   description='A tool to get data Nsights',
   url='https://github.com/cvoscode/Nsight',
   author='cvoscode',
   packages=['nsight'],  #same as name
   install_requires=['dash', 'dash-bootstrap-components', 'pandas','ridgeplot','scikit-learn','waitress','statsmodels','ppscore'], #external packages as dependencies
)
