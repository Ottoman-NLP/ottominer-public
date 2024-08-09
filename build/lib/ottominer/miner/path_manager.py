"""
This module handles file path management in the Miner package.
"""

from pathlib import Path

class PathManager:
    def __init__(self, root_dir: str = None):
        """
        Initialize the PathManager with the root directory.
        
        :param root_dir: str - The root directory of the project. If None, use the cwd.
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.input_dir = self.root_dir / "LLM" / "texture" / "texts"
        self.output_dir = self.root_dir / "LLM" / "texture" / "txts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, *path_parts):
        """
        Construct a path relative to the base directory.
        
        :param path_parts: path components to join.
        :return: Path - The constructed path.
        """
        return self.root_dir.joinpath(*path_parts)
    
    def create_directory(self, dir_path):
        """
        Create a directory if it does not exist.

        :param dir_path: str - The path of the directory to be created.
        """
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directory created at {path}")
        else:
            print(f"Directory already exists at {path}")

    def list_files(self, dir_path, extension=None):
        """
        List files in the directory.

        :param dir_path: str - The path of the directory to list files from.
        :param extension: str - Optional file extension to filter by (e.g., '.txt' or '.pdf').
        :return: list - List of files in the directory.
        """
        path = Path(dir_path)
        if extension:
            return list(path.glob(f'*{extension}'))
        else:
            return list(path.glob('*'))
        
    def list_directories(self, dir_path):
        """
        List directories in the directory.

        :param dir_path: str - The path of the directory to list directories from.
        :return: list - List of directories in the directory.
        """
        path = Path(dir_path)
        return [p for p in path.iterdir() if p.is_dir()]

if __name__ == "__main__":
    root_root = Path(__file__).resolve().parents[3]
    manager = PathManager(root_dir=root_root)
    data_path = manager.get_path("LLM", "texture", "texts")
    manager.create_directory(data_path)
    files = manager.list_files(data_path, extension=".txt")

    for file in files:
        print(f"File: {file}")