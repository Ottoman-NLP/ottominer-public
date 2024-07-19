import os

def load_authors(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            authors = file.read().strip()
        return authors
    except FileNotFoundError:
        return "Authors file not found."
    except UnicodeDecodeError as e:
        return f"Unicode decode error: {str(e)}"

def main():
    metadata = {
        "__author__": load_authors("authors.txt"),
        "__copyright__": "Copyright (c) 2023, Ottoman-NLP",
        "__version__": "1.0.0",
        "__license__": "MIT Open Source License"
    }

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys_version_file = os.path.join(root_dir, "sys_version.txt")

    with open(sys_version_file, 'w') as file:
        file.write(metadata["__version__"])

    return metadata
