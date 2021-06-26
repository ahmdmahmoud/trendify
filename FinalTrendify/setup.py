from setuptools import setup, find_packages
import os


with open('README.md') as f:
    readme = f.read()


reqfile = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(reqfile, 'r') as f:
      requirements = [line.strip() for line in f.readlines()]


setup(
    name='trendify',
    version='0.2',
    description='trends forcasting and detection',
    long_description=readme,
    # packages=find_packages(exclude=('tests', 'docs')),
    packages=find_packages(),
    entry_points={
            'console_scripts': [
                #   'trendify=trendify.trendify:main',
				#   'trendify2=trendify.attention_model_forecast_v1:main',
                  'forecasting=trendify.main.main:main'
            ]
      }
)