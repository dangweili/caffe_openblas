#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N GpuInfor
#$ -o gpu-rate.out
#$ -q NLPR01

hostname
nvidia-smi

