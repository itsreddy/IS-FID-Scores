# IS-FID-Scores
Code to calculate IS and FID scores of GAN generated outputs

This repo is a wrapper around the pyTorch implementation of the FID score calculation code from https://github.com/leognha/PyTorch-FID-score to suit my own project needs. 
Currently works for CIFAR10.

Functions of this repo:
- You can plug-n-play with your models with the generator class in generator.py
- score_utils.py contains the IS-FID score related functions
- dataset_utils contains classes to build custom pytorch datasets over generated images
- calc_cifar1_stats.py contains code to calculate necessary stats for FID to function: will be adding other datasets as needed
