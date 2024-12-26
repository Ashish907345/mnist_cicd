from setuptools import setup, find_packages

setup(
    name="mnist_cicd",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pytest>=6.2.5',
        'numpy>=1.21.0',
        'pillow>=8.3.1',
    ],
) 