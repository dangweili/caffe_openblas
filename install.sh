#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N makecaffe
#$ -o makeoutput.out
#$ -q NLPR01


echo "make clean now..."
make clean
echo "make caffe now..."
make all -j16
echo "make test test caffe now..."
make test
echo "make runtest now..."
make runtest
echo "finish make caffe now.."


