from setuptools import setup, find_packages

setup(
    name='ottominer',
    version='0.1.2',
    description='An NLP mining and extraction package for Ottoman documents',
    author='rekurrenzk',
    author_email='rekurrenzk@proton.me',
    packages=find_packages(),
    install_requires=[
        'PyMuPDF',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
