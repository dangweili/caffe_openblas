#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N GpuInfor
#$ -o gpu-rate
#$ -q NLPR01

hostname
nvidia-smi

