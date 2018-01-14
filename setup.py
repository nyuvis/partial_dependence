# -*- coding: utf-8 -*-
"""
Created on 2018-01-14
"""
from setuptools import setup

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# NOTE! steps to distribute:
#$ python setup.py sdist bdist_wheel
#$ twine upload dist/... <- here be the new version!

setup(
    name='partial_dependence',
    version='0.0.1',
    description='TODO',
    long_description=long_description,
    url='https://github.com/nyuvis/partial_dependence',
    author='Paolo Tamagnini',
    author_email='paolotamag@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    keywords='TODO',
    py_modules=['partial_dependence'],
    install_requires=[],
)
