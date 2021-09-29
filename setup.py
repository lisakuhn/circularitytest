#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    name='circularitytest',
    description='Python implementation of circularitytest',
    author='Lisa Kuhn',
    url='https://github.com/liskuhn/circularitytest',
    license='',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
        ],
    }
)