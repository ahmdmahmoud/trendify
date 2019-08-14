# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

reqfile = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(reqfile, 'r') as f:
      requirements = [line.strip() for line in f.readlines()]
      
setup(
    name='trendify',
    version='0.1',
    description='Attention models for trends forcasting',
    long_description=readme,
    author='Ahmed Saleh',
    author_email='ahmed.mahmoud@fcih.net',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points={
            'console_scripts': [
                  'trendify=trendify.trendify:main',
				  'trendify2=trendify.attention_model_forecast_v1:main',
            ]
      }
)