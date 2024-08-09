from setuptools import setup, find_packages

setup(
    name='ottoman_miner',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        #will be provided later
    ],
    tests_require=[
        'unittest2'
    ],
    test_suite = 'tests',
    description = 'A roboust text mining package for Ottoman Turkish texts in NLP tasks',
    author = 'Fatih Burak Karagöz',
    author_email='felixfelicies@protonmail.ch',
    license='MIT',

)
