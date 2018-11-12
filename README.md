# Reliable and Interpretable AI Project

This project uses `git lfs`. Please refer to [their website](https://git-lfs.github.com/) for details.

## Running the Project

The project is hosted on a docker image. This might not be 100% stable.

In order to build the docker image please run:

```sh
docker build . -t relai
```

Then use the following command to run an analysis:

```sh
docker run relai ../mnist_nets/<net_name>.txt ../mnist_images/<image>.txt <epislon>
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

- `imgn` where `n` ranges from 0 to 99 inclusive.

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
