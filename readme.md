<p align="center">
  <img src="items/readme_logo.png" alt="Ottoman NLP Toolkit Logo" width="200" style="margin-right: 20px;"/>
  <img src="items/boun.png" alt="Boğaziçi University Logo" width="200"/>
</p>
<p align="center">
  <em>Ottoman NLP</em>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Boğaziçi BUCOLIN Lab</em>
</p>


# Ottoman Miner

A comprehensive text mining and analysis toolkit tailored for Ottoman archival documents, developed by the BUCOLIN Lab at Boğaziçi University.

## Overview

Ottoman Miner is an integral component of the Ottoman NLP project, engineered to transform the processing and analysis of Ottoman Turkish texts. This toolkit effectively addresses the inherent challenges associated with text extraction and analysis from historical Ottoman documents, equipping researchers with advanced tools for digital Ottoman studies.

## Features

- **Advanced PDF Processing**: Specialized extraction techniques optimized for Ottoman archival documents.
- **Intelligent Text Mining**: Pattern recognition algorithms tailored for Ottoman Turkish.
- **Parallel Processing**: Efficiently manages large historical collections through concurrent processing.
- **Semantic Analysis**: Comprehensive tools designed to interpret Ottoman linguistic patterns.
- **Progress Monitoring**: Real-time tracking of processing status with detailed logging capabilities.
- **Resource Optimization**: Intelligent management of system resources to enhance performance.
- **Configurable Workflows**: Flexible processing pipelines adaptable to various research needs.

## Installation

To install Ottoman Miner, execute the following commands:

```bash
git clone https://github.com/Ottoman-NLP/ottominer-public.git
cd ottominer-public
pip install -e .
```
## Quick Start

Begin utilizing Ottoman Miner with the following Python script:

``` 
python.exe


from ottominer.core import Environment
from ottominer.analyzer import semantic

env = Environment()

analyzer = semantic.SemanticAnalyzer()
results = analyzer.process("WILL/BE/UPDATED.pdf")
```

## Project Structure

ottominer/
├── analyzer/    # Components for Ottoman text analysis
├── cli/         # Command-line interface tools
├── core/        # Core functionalities of the toolkit
├── extractors/  # Document extraction utilities
├── utils/       # Auxiliary utility functions
└── tests/       # Comprehensive test suite

## Configuration

Ottoman Miner can be configured through the following methods:

- Command-Line Arguments: Customize settings directly via the CLI.
- Configuration File: Modify settings in ~/.ottominer/config.json.
- Environment Variables: Set environment-specific variables for configuration.

## Development and Testing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute and test the project.

## Contributing

Contributions from researchers and developers are highly encouraged. To contribute:

    Fork the repository.
    Create a feature branch: `git checkout -b feature-name`
    Commit your changes: `git commit -am 'Add new feature'`
    Push to the branch: `git push origin feature-name`
    Submit a pull request.

## Citation

If you use Ottoman Miner in your research, please cite the following paper:

```
@inproceedings{karagoz2024towards,
  title={Towards a Clean Text Corpus for Ottoman Turkish},
  author={Karagöz, Fatih Burak and Doğan, Berat and Özateş, Şaziye Betül},
  booktitle={Proceedings of the First Workshop on Natural Language Processing for Turkic Languages (SIGTURK 2024)},
  pages={62},
  year={2024},
  month={August},
  address={Bangkok, Thailand},
  publisher={ACL}
}
```
Acknowledgments

    BUCOLIN Lab, Boğaziçi University
    Digital Ottoman Studies Initiative
    Contributors and Researchers in Ottoman studies
    Open-Source NLP Community

Contact

    Project Manager: Şaziye Betül Özateş
    Project Lead: Fatih Burak Karagöz
    Repository: Ottoman-NLP/ottominer-public
    Organization: Ottoman-NLP


© 2023 Ottoman NLP Project, Boğaziçi University. All rights reserved.