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
    return metadata
