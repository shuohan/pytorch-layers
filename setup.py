# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

scripts = glob('scripts/*')
command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='pytorch-layers',
      version=version,
      description='Factory methods to create PyTorch layers.',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      scripts=scripts,
      install_requires=['torch==2.2.0'],
      packages=['pytorch_layers'])
