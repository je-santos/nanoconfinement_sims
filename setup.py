#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:49:25 2019

@author: jesantos
"""

import os
import sys
from distutils.util import convert_path
from setuptools import find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#sys.path.append(os.getcwd())


setup(
    name='MD_trainingset_creator',
    description = 'Interface for creating geometries and running MD \
    Simulation',
    #version=main_['__version__'],
    classifiers=[
        'Development Status :: 0 - Underdevelopment',
        'Programming Language :: Python',
        'Topic :: Scientific',
    ],
    packages = find_packages(),
    install_requires=[
        'numpy>=1.16',
        'scipy>=1.1',
        'matplotlib',
        'pandas',
        'moltemplate',
        'packmol'
        ],
    author='Javier E. Santos',
    author_email='jesantos@utexas.edu',
    )