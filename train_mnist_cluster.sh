#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -N caffe_mnist
#$ -o ./outresult
#$ -e ./outerror
#$ -q NLPR01

echo "creating mnist dataset"
./examples/mnist/create_mnist.sh

echo "train the model"
./examples/mnist/train_lenet.sh

echo "finishing training"
