import sys
from setuptools import setup

# Written according to the docs at
# https://packaging.python.org/en/latest/distributing.html

if sys.version_info[0] < 3:
    sys.exit('This script requires python 3.0 or higher to run.')

setup(
    name='psopy',
    description='A Particle Swarm Optimizer',
    version='0.1.3',
    url='https://github.com/NDari/psopy',
    author='Naseer Dari',
    author_email='naseerdari01@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    packages=['psopy'],
    install_requires=['numpy']
)
