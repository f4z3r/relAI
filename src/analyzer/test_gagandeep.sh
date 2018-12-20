#!/usr/bin/env bash

echo "Linear programming 6x100 epsilon 0.01 image 45"
./test.sh mnist_relu_6_100 45 0.01

echo "Linear programming 6x50 epsilon 0.1 image 0"
./test.sh mnist_relu_6_50 0 0.1

echo "Linear programming 4x1024 epsilon 0.001 image 0"
./test.sh mnist_relu_4_1024 0 0.001

echo "Linear programming 9x200 epsilon 0.01 image 0"
./test.sh mnist_relu_9_200 0 0.01

echo "Back propagation 4x1024 epsilon 0.003 image 0, capacity 60"
./test.sh mnist_relu_4_1024 0 0.003

echo "Back propagation 4x1024 epsilon 0.006 image 0, capacity 40"
./test.sh mnist_relu_4_1024 0 0.006
