# Reliable and Interpretable AI Project

This project uses `git lfs`. Please refer to
[their website](https://git-lfs.github.com/) for details.

## Running the Project

### Virtual Machine

To run the project in the virtual machine, simply clone the `git` repo
somewhere into the VM. Then run `setup_vm.sh` from within the cloned repo.
This should provide you with the functionality provided below, but you will
need to substitute `docker run` with `test.sh` and run the commands within
`/home/riai2018/analyzer/`.

Note that you might also require to run `source setup_gurobi.sh` in order to
setup environment variables to make gurobi work properly. If you encounter an
error indicating the following:

```sh
ImportError: libgurobi81.so: cannot open shared object file: No such file or directory.
```

It indicates that you need to source `setup_gurobi.sh`.

### Docker

The project is hosted on a docker image. This can be used if the gurobi solver
is not required or you have a valid license to inject into the docker container.

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

Running tests allows to see aggregate statistics for more than a single run. A
sample output of running some tests is:

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

will run 100 tests over all images for the `mnist_relu_3_10` neural net with an
epsilon value of `0.001`.

### Running full Tests

The docker can also be passed the `test` argument to fully test the entire
analyzer over many experiments:

```sh
docker run -v $(pwd)/src/analyzer/:/home/riai2018/analyzer relai test
```

## Project Description

- Write a generic analyser to verify the robustness of any feed-forward network
  using ReLU activation functions.
- Find robustness by looking for adversarial examples within an L-inf norm
- The verifier should leverage interval domains and linear programming. Interval
  domains are fast but imprecise, linear programming is precise but quite slow.

The VM given already contains a fully working interval domain analyser.

Note that the docker image should work as well.

### Task

Improve the precision by modifying the analyse function. Do this by augmenting
the analyser using linear programming.

### Grading

There will be a timeout limit at 7 minutes. You get points for correctly
verified solutions.

The inputs given to the analyser will satisfy the following properties:

- NN that have at least 3 and at most 9 fully connected FF layers. Each layer
  consists of 20 to 2000 neurons.
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

In order to add full log information use the following in this file:

```
<details>
  <summary>network_name(espilon)</summary>

  log file here

</details>
```


### Full Interval Propagation - implemented

Perform full interval propagation on the network. This is nearly instantaneous
on small networks and takes at most a few seconds on the largest networks. The
precision strongly degrades as the networks get both deeper and wider.

### Full Linear Programming - implemented

Perform full linear programming on the network. This only takes a few seconds on
the smaller networks, but can take many several minutes and up to hours on the
very large networks.

There seems to be massive time discrepancies between images that can and cannot
be verified on the same network size. For instance, on some networks, an image
can get verified in about 40 seconds using full linear programming, but another
image can take up to 40 _minutes_ to be rejected on the very same network with
the very same epsilon size.

#### Notes

<details>
  <summary>3x10 (0.1)</summary>

  [log file](./logs/3_10_0.1_full_lp.log)

</details>

<details>
  <summary>3x20 (0.1)</summary>

  [log file](./logs/3_20_0.1_full_lp.log)

</details>

<details>
  <summary>3x50 (0.1)</summary>

  [log file](./logs/3_50_0.1_full_lp.log)

</details>

<details>
  <summary>6x20 (0.1)</summary>

  [log file](./logs/6_20_0.1_full_lp.log)

</details>

<details>
  <summary>6x50 (0.1)</summary>

  [log file](./logs/6_50_0.1_full_lp.log)

</details>

<details>
  <summary>6x100 (0.1)</summary>

  [log file](./logs/6_100_0.1_full_lp.log)

</details>

<details>
  <summary>6x200 (0.1) -- partial</summary>

  [log file](./logs/6_200_0.1_full_lp_partial.log)

</details>

### Neuronwise Heuristic

Produce a scoring mechanism that scores a neuron based on importance and thus
chooses the best neurons on which to perform linear programming per layer.

#### Weight Scores - implemented

This heuristic looks at a neuron's outgoing weights to determine its score.
Moreover, this score is combined with the output bounds of the neuron. In the
current implementation, this is actually slower than simply performing linear
programming on the entire network _for deep networks_. This might be due to
inefficient sorting implementations.


### Window Linear Programming - implemented

Perform linear programming on a window of layers of the network. In this
heuristic, only a few consecutive layers are modelled for linear programming.
This opposes the full linear program that encodes a model all the way back to
the input layer. This partial model is then moved across the network as if it
were a window. This seems to have decent precision, but the time efficiency of
this is not yet tested.

This currently seems to be by far the most promising technique. It verifies with
very high precision but takes a fraction of the time of full linear programming.

#### Notes

Window linear programming only makes real sense to perform on the networks
having 9 layers. On networks with less layers, the window size must be nearly
the entire network, hence full linear programming is a better choice.

<details>
  <summary>9x100</summary>

  When running this heuristic on the `mnist_relu_9_100` network with an epsilon
  of 0.01, an overall precision of 78% is reached (with a `window_size` of `4`).
  The ground thruth of this data set is not available but comparably to the
  available data, this seems to be extremely close to the ground truth. In
  detail:

  - network: `mnist_relu_9_100`
  - epsilon: 0.01
  - 78% overall precision
  - about 15.5 seconds per image on average
  - fastest verification: image 82 in 7 seconds
  - slowest verification: image 37 in 19 seconds
  - fastest fail: image 73/38 in 1 second
  - slowest fail: image 43 in 79 seconds
  - speedup compared to full linear programming seems to be between factor 1.5
    and factor 2

  This compares to the speed of full LP.

</details>


### Partial Linear Programming - implemented

Perform linear programming on the first few layers of the network, then resort
to interval propagation.

### Incomplete Linear Programming - implemented

Perform linear programming by propagating the bounds of elina, and performing
optimisation steps only on the very last layer, as opposed to on every neuron
of the network. This has extremely good performance, but severely lacks
precision on larger networks.

### Back-Propagation - implemented

Perform interval propagation to have a general idea of the intervals each
neuron can take. Then, starting at the output layer:

1. For each neuron `n`:
   - Check which neurons can affect its value the most. This is performed by
     checking the possible interval size of each incoming neuron `m` and
     multiplying it by the weight between `m` and `n`. This gives a general
     extimation how much `m` affect the output interval of `n`.
   - Based on the scores computed in the previous step, take the highest
     `capacity` neurons that affect neuron `n` and store them.
2. Compute the union of all high impact sets returned.
3. Repeat from step 1 using the previous layer (layer that so far contained the
   `m` neurons), but only check for high impact neurons in the list returned
   from step 2. Back-propagate like this until the input layer is reached.
4. We now have a list of high impact neuron sets for each layer:
   - Starting at the first hidden layer, compute linear programming for the high
     impact neurons on this layer, and perform interval propagation on all other
     neurons of this layer.

By default, all neurons in the output layer are considered "high impact", hence
linear programming will always be performed on all output neurons.

#### Notes

This strategy seems to reduce runtime quite significantly compared to linear
programming and should mostly be used on the 4x1024 network. It seems that a
good estimate for the `capacity` variable is the number of neurons in the hidden
layer divided by the number of nuerons in the output layer (hence about 100 for
the 4x1024 network). The time efficiency is not yet properly investigated.

<details>
  <summary>4x1024 (0.001) -- old strategy</summary>

  [log file](./logs/4_1024_0.001_backprop.log)

</details>


Now a new technique is adopted. Fist of all, not all output neurons are
necessarily chosen for the back-propagation. If the neuron satisfies the bounds
anyways, there is no need for backpropagating the neuron. Moreover, output
neurons that have larger differences to the bound satisfaction are awarded more
high impact neurons to back-propagate.

<details>
  <summary>6x50 (0.01) capacity=1000 -- sanity check</summary>

  [log file](./logs/6_50_0.01_back_prop_1000.log)

</details>

<details>
  <summary>General Performance difference to LP</summary>

  This was run using back propagation with capacity 20 first. Then full LP. As
  one can see, there is close to no performance difference but full LP is more
  precise.

  ```
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_6_200 0 0.01
  Academic license - for non-commercial use only
  can not be verified
  analysis time:  39.93522930145264  seconds
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_6_200 0 0.01
  Academic license - for non-commercial use only
  verified
  analysis time:  41.21198058128357  seconds
  ```

  On larger networks, again fist run with 20 capacity, then full LP. Here both
  manage to verify the image, but there is nearly no performance difference:

  ```
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_9_200 0 0.01
  Academic license - for non-commercial use only
  verified
  analysis time:  64.45668196678162  seconds
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_9_200 0 0.01
  Academic license - for non-commercial use only
  verified
  analysis time:  67.49932312965393  seconds
  ```

  On the very large networks, there seems to be a performance difference. Here
  the first run used back-propagation with a capacity of 100, the second run
  used full LP:

  ```
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.001
  Academic license - for non-commercial use only
  verified
  analysis time:  251.8695423603058  seconds
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.001
  Academic license - for non-commercial use only
  verified
  analysis time:  553.6867568492889  seconds
  ```

  However, this quickly becomes less performant as epsilon is increased:

  ```
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.003
  Academic license - for non-commercial use only
  can not be verified
  analysis time:  674.2426040172577  seconds
  ```

  Moreover, dopping the capacity does not seem to affect the runtime very much.
  Here is an example running with capacity 70:

  ```
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.001
  Academic license - for non-commercial use only
  verified
  analysis time:  206.27014303207397  seconds
  ```

  And with capacity 40:

  ```
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.001
  Academic license - for non-commercial use only
  705
  502
  390
  273
  10
  can not be verified

  analysis time:  135.3421106338501  seconds
  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.003
  Academic license - for non-commercial use only
  688
  490
  376
  302
  10
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  [64.7209760953626, 82.36587988210604, 84.32325623763094, 92.60535297660537, 70.72618522123024, 77.13598857657838, 50.43105982922593, 97.92398053668056, 79.70707780919288, 90.43177901191099]
  can not be verified
  analysis time:  182.88847398757935  seconds

  riai2018@riai2018-VirtualBox:~/analyzer$ ./test.sh mnist_relu_4_1024 0 0.006
  Academic license - for non-commercial use only
  705
  476
  452
  307
  10
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  [176.30024402144548, 213.72371651572283, 220.90109864400554, 228.67297077325838, 196.7929647217841, 213.0581433462186, 172.36260804621605, 223.39306625460574, 213.2329928135486, 230.86499535159382]
  can not be verified
  analysis time:  338.62040972709656  seconds
  ```

</details>


### Random Tests

```
riai2018@riai2018-VirtualBox:~/analyzer$ ./test_gagandeep.sh
Linear programming 6x100 epsilon 0.01 image 45
Academic license - for non-commercial use only
[0, 0, 0, 2.3361461064725524, 0, 6.3224056798203545, 0, 0, 0, 0.722075323751903]
[0.5970363867756227, 0, 0, 4.1038713155896485, 0, 8.967534078296243, 0.03764919380555115, 0, 0.19674928596310587, 1.6752748318054318]
verified
analysis time:  10.376429319381714  seconds

Linear programming 6x50 epsilon 0.1 image 0
Academic license - for non-commercial use only
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[38.8877242608964, 42.357221518432155, 43.45314080067412, 60.88137027505754, 43.79019047143408, 51.454892056012376, 45.78950815787175, 56.45468642634735, 48.68513993725941, 56.85771677837251]
can not be verified
analysis time:  11.348645448684692  seconds

Linear programming 4x1024 epsilon 0.001 image 0
Academic license - for non-commercial use only
[0, 4.763251885992395, 1.270932795649884, 10.04945198832089, 0, 0, 0, 27.181769117024185, 0, 4.833550044432829]
[0, 4.9702055105947185, 1.533344244491642, 10.381947141851763, 0, 0, 0, 27.587246729362196, 0, 5.158394982029161]
verified
analysis time:  525.1957831382751  seconds
Linear programming 9x200 epsilon 0.01 image 0

Academic license - for non-commercial use only
[0, 0, 0, 1.7902444960401407, 0, 0, 0, 10.387748038956925, 0, 3.561284626930758]
[0, 0, 0.01090494940429658, 2.238246453070047, 0, 0, 0, 12.79869169211072, 0, 4.813640443725761]
verified
analysis time:  63.592748403549194  seconds
Back propagation 4x1024 epsilon 0.003 image 0, capacity 60

Academic license - for non-commercial use only
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[41.27759275560801, 54.82434326646063, 56.08237267395752, 64.16462292536244, 43.12641212705333, 48.37556473060915, 25.40719744548523, 71.63077226839238, 51.62828037429907, 62.02249357059396]
can not be verified
analysis time:  369.7813322544098  seconds
Back propagation 4x1024 epsilon 0.006 image 0, capacity 40

Academic license - for non-commercial use only
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[176.30024402144548, 213.72371651572283, 220.90109864400554, 228.67297077325838, 196.7929647217841, 213.0581433462186, 172.36260804621605, 223.39306625460574, 213.2329928135486, 230.86499535159382]
can not be verified
analysis time:  416.097154378891  seconds
```
