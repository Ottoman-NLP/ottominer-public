<p align="center">
  <img src="items/readme_logo.png" alt="Ottoman NLP Toolkit Logo" width="200" style="margin-right: 20px;"/>
  <img src="items/boun.png" alt="Boğaziçi University Logo" width="200"/>
</p>
<p align="center">
  <em>Ottoman NLP</em>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em>Boğaziçi BUCOLIN Lab</em>
</p>

# Ottoman NLP

## Research Objectives

This project aims to develop a state-of-the-art model for normalizing Ottoman Turkish texts. Using advanced Natural Language Processing (NLP) techniques, we transform non-standardized or noisy Ottoman Turkish text into a normalized form, facilitating further analysis and preservation of historical documents.

## Project Overview

The OttoMiner Toolkit is a cutting-edge computational linguistics project designed to revolutionize the processing and analysis of Ottoman Turkish texts. Developed by the BUCOLIN Lab at Boğaziçi University, this toolkit addresses the unique challenges posed by the Ottoman language's complex script and historical variations.

Our toolkit consists of three main processing phases:

### 1. ottominer

- A static parsing module utilizing accessible PDF extraction libraries (Fitz, Pdfplumber)
- Reformats unstructured texts using a modular regex detection mechanism
- Aims to create contextual texts while minimizing sentence structure errors and noise patterns
- Fully developed but requires updates for:
  - Cross-platform applicability
  - More robust corpus cleaning proficiency

### 2. OttoRecognition

- Currently in development
- Focuses on creating an OCR-based machine learning algorithm to:
  - Extract original Ottoman texts
  - Automate transliteration into latinized Turkish
- Goals:
  - Enable uniform formatting of documents
  - Increase available textual data (latinized versions of Ottoman texts are currently limited)

### 3. Machine Translation Architecture

To complement our static parsing package and address unforeseen errors, we're developing a Machine Translation (MT) architecture. This framework will help create more aligned data resources:


```
         ┌─────────┐                  ┌─────────┐
Input -> │ Encoder │ -> Context ->    │ Decoder │ -> Output
         └─────────┘                  └─────────┘
             ↑                            ↑
        Embedding                    Embedding
```

## Repository Guidelines

### Corpus-Texts

- Available at: [https://github.com/Ottoman-NLP/corpus-texts](https://github.com/Ottoman-NLP/corpus-texts)
- Contains:
  - Clean text resources for further training
  - Datasets for text-mining and OCR detection
- For more information, refer to the README file in the corpus-texts repository

### Parsing Package

To use ottominer for extracting PDF documents into TXT format:

1. Navigate to `ottominer-public/ottominer/HERE_ILL_PUT_EXE_FILE.exe`
2. Use your custom PDF document folder as the path address
3. The extraction will create a new folder with the same name as the input folder, but with a .txt extension for the extracted texts

### Project-Based Documents & Texts

- Located in the `corpus-texts` repository
- For further details and acknowledgments, please read `/corpus-texts/README.md`


## Technical Specifications

- **Programming Languages**: Python 3.8+, Rust, C++
- **Machine Learning Frameworks**: PyTorch, TensorFlow, T5, Seq2Seq
- **Database**: plain-texts, csv and json for labelling
- **Front-end**: React.js with D3.js for visualizations

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Ottoman-NLP/ottominer-public
   cd ottominer-public
   ```

2. Set up the environment:
   ```bash
   python -m venv .venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Configure the database:
   ```bash
   python scripts/setup_database.py
   ```

4. Launch the application:
   ```bash
   python -m GUI.tk_main
   ```

## Usage Guidelines

Refer to our [comprehensive documentation](https://ottoman-nlp.readthedocs.io) for detailed usage instructions, API references, and best practices.

## Contributing

We welcome contributions from researchers and developers. Please consult our [Contribution Guidelines](CONTRIBUTING.md) for more information on how to submit pull requests, report issues, or suggest enhancements.

## Citing This Work

If you use the Ottoman NLP Toolkit in your research, please cite our paper:
<pre>
@inproceedings{karagoz2024towards,
title={Towards a Clean Text Corpus for Ottoman Turkish},
author={Karag{"o}z, Fatih Burak and Do{\u{g}}an, Berat and {"O}zate{\c{s}}, {\c{S}}aziye Bet{"u}l},
booktitle={Proceedings of the First Workshop on Natural Language Processing for Turkic Languages (SIGTURK 2024)},
pages={62},
year={2024},
month={August},
address={[Bangkok, Thailand]},
publisher={[ACL]}
}
</pre>

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

© 2023 Ottoman NLP Project, Boğaziçi University. All rights reserved.