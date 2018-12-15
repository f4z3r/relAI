# Reliable and Interpretable AI Project

This project uses `git lfs`. Please refer to [their website](https://git-lfs.github.com/) for details.

## Running the Project

### Virtual Machine

To run the project in the virtual machine, simply clone the `git` repo somewhere into the VM. Then run `setup_vm.sh` from within the cloned repo. This should provide you with the functionality provided below, but you will need to substitute `docker run` with `test.sh` and run the commands within `/home/riai2018/analyzer/`.

Note that you might also require to run `source setup_gurobi.sh` in order to setup environment variables to make gurobi work properly. If you encounter an error indicating the following:

```sh
ImportError: libgurobi81.so: cannot open shared object file: No such file or directory.
```

It indicates that you need to source `setup_gurobi.sh`.

### Docker

The project is hosted on a docker image. This can be used if the gurobi solver is not required or you have a valid license to inject into the docker container.

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

### Full Interval Propagation - implemented

Perform full interval propagation on the network. This is nearly instantaneous on small networks and takes at most a few seconds on the largest networks. The precision strongly degrades as the networks get both deeper and wider.

### Full Linear Programming - implemented

Perform full linear programming on the network. This only takes a few seconds on the smaller networks, but can take many several minutes and up to hours on the very large networks.

There seems to be massive time discrepancies between images that can and cannot be verified on the same network size. For instance, on some networks, an image can get verified in about 40 seconds using full linear programming, but another image can take up to 40 _minutes_ to be rejected on the very same network with the very same epsilon size.

### Neuronwise Heuristic

Produce a scoring mechanism that scores a neuron based on importance and thus chooses the best neurons on which to perform linear programming per layer.

#### Weight Scores - implemented

This heuristic looks at a neuron's outgoing weights to determine its score. Moreover, this score is combined with the output bounds of the neuron. In the current implementation, this is actually slower than simply performing linear programming on the entire network _for deep networks_. This might be due to inefficient sorting implementations.


### Window Linear Programming - implemented

Perform linear programming on a window of layers of the network. In this heuristic, only a few consecutive layers are modelled for linear programming. This opposes the full linear program that encodes a model all the way back to the input layer. This partial model is then moved across the network as if it were a window. This seems to have decent precision, but the time efficiency of this is not yet tested.

This currently seems to be by far the most promising technique. It verifies with very high precision but takes a fraction of the time of full linear programming.

#### Notes

Window linear programming only makes real sense to perform on the networks having 9 layers. On networks with less layers, the window size must be nearly the entire network, hence full linear programming is a better choice.

##### 9x100

When running this heuristic on the `mnist_relu_9_100` network with an epsilon of 0.01, an overall precision of 78% is reached (with a `window_size` of `4`). The ground thruth of this data set is not available but comparably to the available data, this seems to be extremely close to the ground truth. In the detail:

- network: `mnist_relu_9_100`
- epsilon: 0.01
- 78% overall precision
- about 15.5 seconds per image on average
- fastest verification: image 82 in 7 seconds
- slowest verification: image 37 in 19 seconds
- fastest fail: image 73/38 in 1 second
- slowest fail: image 43 in 79 seconds
- speedup compared to full linear programming seems to be between factor 1.5 and factor 2

### Partial Linear Programming - implemented

Perform linear programming on the first few layers of the network, then resort to interval propagation.

### Incomplete Linear Programming - implemented

Perform linear programming by propagating the bounds of elina, and performing optimisation steps only on the very last layer, as opposed to on every neuron of the network. This has extremely good performance, but severely lacks precision on larger networks.
