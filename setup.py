from setuptools import setup, find_packages

setup(
    name="ottominer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'rich>=13.3.0',  # Specific version for RichHelpFormatter
        'pytest>=6.0.0',
        'typing-extensions>=4.0.0',
        'textual>=0.1.0',
    ],
    python_requires='>=3.8',
) 