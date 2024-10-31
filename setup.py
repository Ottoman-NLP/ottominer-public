from setuptools import setup, find_packages

setup(
    name="ottominer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'rich>=10.0.0',
        'pytest>=7.0.0',
        'pyyaml>=6.0.0',
        'psutil>=5.9.0',
        'pymupdf4llm>=0.0.17',
        'reportlab>=4.0.0',
    ],
    python_requires='>=3.8',
    test_suite='tests',
) 