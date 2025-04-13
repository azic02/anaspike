from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='anaspike',
    version='0.2.0',
    packages=find_packages(),
    install_requires=requirements,
    description='An analysis package for modular spiking network models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/azic02/anaspike',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

