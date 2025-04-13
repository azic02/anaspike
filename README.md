# anaspike

## Installation
Run
```
pip install git+https://github.com/azic02/anaspike.git
```
   
## Run tests

### unit tests
Run
```
python -m unittest
```
in the root directory of the repository.

### functional tests
Functional tests are contained in the [text notebook](https://jupytext.readthedocs.io/en/latest/text-notebooks.html)
`./tests/functional_tests.py`. One must provide a `.h5` file containing test
simulation data compatible with the `anaspike.dataclasses.SimulationData`
class.

