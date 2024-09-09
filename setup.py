from setuptools import setup, find_packages

setup(
    name='ottominer',
    version='0.2.5',
    description='An NLP mining and extraction package for Ottoman documents',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='rekurrenzk',
    author_email='rekurrenzk@proton.me',
    url='https://github.com/Ottoman-NLP/ottominer-public',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        line.strip() for line in open('requirements.txt')
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='nlp ottoman mining extraction',
)
