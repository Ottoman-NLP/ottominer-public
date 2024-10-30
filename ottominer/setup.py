from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        # Import and run after installation
        from ottominer import setup
        setup()

setup(
    name='ottominer',
    version='0.1',
    packages=find_packages(),
    cmdclass={
        'install': PostInstallCommand,
    },
    entry_points={
        'console_scripts': [
            'ottominer=ottominer.main:main',
        ],
    },
)