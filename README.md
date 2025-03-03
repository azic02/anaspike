# anaspike

## Installation
Requires NEST to be installed.

### pre-installed NEST
If pyNEST (and the NEST simulator) is already installed, run 
```
pip install .
```
inside the cloned repository to install the package and further required
dependencies
  
### Docker setup
If NEST is not installed, follow the following instructions to be able to run
inside a Docker container
1. Install [Docker Engine](https://docs.docker.com/engine/install/)
2. Clone this repository
3. Change directory into the cloned repository and create a Docker image from
   the provided Dockerfile:
   ```
   docker build -t anaspike .
   ```
4. Create a Docker container from the installed image:
   ```
   docker run -it --mount type=bind,source="$(pwd)",target=/opt/data -p 5000:5000 --name anaspike anaspike
   ```
5. Inside the container, run
   ```
   pip install .
   ```

Refer to [Docker setup usage](#docker-setup-1)
   
## Usage
As of now, the main functionality is provided by the analysis scripts inside
the root directory of the repository.
  
### pre-installed NEST
After following the [instructions for installation](#pre-installed-nest), one should be able to just
run the scripts as normal.

### Docker setup
To exit the container, run
```
exit
```
To start it up again, run
```
docker start -i anaspike
```
For further information on the management of Docker containers and images,
refer to their documentation.

To run jupyter-notebook inside a Docker container, run
```
jupyter-notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root
```
and open the given URL in a browser.
  

## Run tests

### unit tests
Run
```
python -m unittest
```
in the root directory of the repository.

### functional tests
Run
```
./run_functional_tests.sh
```
in the root directory of the repository. The generated output can be found in
the `/tests/data` and `/tests/plots` directories.

