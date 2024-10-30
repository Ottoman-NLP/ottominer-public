# Contributing to Ottoman Miner

We love your input! We want to make contributing to Ottoman Miner as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Development Process
We use Github Flow, so all code changes happen through pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

1. Clone your fork: 

```
bash
git clone https://github.com/your-username/ottominer-public.git
cd ottominer-public
```
2. Create a virtual environment:

```
bash
python -m venv venv
source .venv/bin/activate <---#---> On Windows: .venv\Scripts\activate
```
3. Install dependencies:

```
bash
pip install -r requirements.txt
pip install -e .
```
4. Setup pre-commit hooks:

```
bash
pre-commit install
```

## Code Style
- We use `black` for Python formatting
- We use `isort` for import sorting
- We use `flake8` for linting

## Testing
Run the test suite:

```
bash
python -m pytest
```

With coverage:

```
bash
python -m pytest --cov=ottominer
```

## License
By contributing, you agree that your contributions will be licensed under its MIT License.


.gitignore:

```
htmlcov/
.pytest_cache/
.tox/
```

5. pre-commit-config.yaml:

```
ottominer-public/.pre-commit-config.yaml
```

