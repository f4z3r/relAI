# Reliable and Interpretable AI Project

This project uses `git lfs`. Please refer to [their website](https://git-lfs.github.com/) for details.

## Running the Project

The project is hosted on a docker image.

In order to build the docker image please run:

```sh
docker build . -t relai
```

from the project root. Then use the following command to run an analysis:

```sh
docker run -v $(pwd)/src/analyzer/:/home/riai2018/analyzer relai <net_name> <image> <epsilon>
```

Possible values for MNIST nets are:

- `mnist_relu_3_10`
- `mnist_relu_3_20`
- `mnist_relu_3_50`
- `mnist_relu_4_1024`
- `mnist_relu_6_20`
- `mnist_relu_6_50`
- `mnist_relu_6_100`
- `mnist_relu_6_200`
- `mnist_relu_9_100`
- `mnist_relu_9_200`

Possible values for the MNIST images are:

- Range from 0 to 99 inclusive.

An example command would be:

```sh
docker run -v $(pwd)/src/analyzer/:/home/riai2018/analyzer relai mnist_relu_3_10 2 0.1
```

called from the project root.

## Tests

Running tests allows to see aggregate statistics for more than a single run. A sample output of running some tests is:

```
Testing net 'mnist_relu_3_10' for epsilon 0.001
Total tests:  100
Passed tests: 93
Percentage:   93.00%
Total time:   28s
Time/test:    .28s
```

### Running Partial Tests

A single network can be run on all images for a single epsilon value by using:

```sh
docker run -v $(pwd)/src/analyzer/:/home/riai2018/analyzer relai <net_name> <epsilon>
```

Hence running:

```sh
docker run -v $(pwd)/src/analyzer/:/home/riai2018/analyzer relai mnist_relu_3_10 0.001
```

will run 100 tests over all images for the `mnist_relu_3_10` neural net with an epsilon value of `0.001`.

### Running full Tests

The docker can also be passed the `test` argument to fully test the entire analyzer over many experiments:

```sh
docker run -v $(pwd)/src/analyzer/:/home/riai2018/analyzer relai test
```

## Project Description

- Write a generic analyser to verify the robustness of any feed-forward network using ReLU activation functions.
- Find robustness by looking for adversarial examples within an L-inf norm
- The verifier should leverage interval domains and linear programming. Interval domains are fast but imprecise, linear programming is precise but quite slow.

The VM given already contains a fully working interval domain analyser.

Note that the docker image should work as well.

### Task

Improve the precision by modifying the analyse function. Do this by augmenting the analyser using linear programming.

### Grading

There will be a timeout limit at 7 minutes. You get points for correctly verified solutions.

The inputs given to the analyser will satisfy the following properties:

- NN that have at least 3 and at most 9 fully connected FF layers. Each layer consists of 20 to 2000 neurons.
- images from the MNIST dataset
- epsilon values ranging between 0.005 and 0.1

### Constraints

- Implementation must be in python.
- Interval domain must use the interface of ELINA.
- Linear Solver should use the Gurobi framework.
- No abstract domains allowed.
- No additional libraries

### Deadline

20th of December

## Proposed Solution

TODO
