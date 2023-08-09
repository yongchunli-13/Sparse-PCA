# Exact and Approximation Algorithms for Sparse PCA (SPCA)

This project contains the implementations of exact algorithms, approximation algorithms, and continuous relaxations for SPCA in our paper "Exact and Approximation Algorithms for Sparse PCA": http://www.optimization-online.org/DB_FILE/2020/05/7802.pdf.

## Exact algorithms
We propose three equivalent mixed-integer convex programs for SPCA, denoted by MISDP (6), MISDP (15), and MILP (22) in our paper. To be specific, we design a branch-and-cut algorithm to solve MISDP (6), where the implementation can be found in "", repsectively
Specifically, the "Exact algorithms" folder contains the 
we compute a lower bound and an upper bound of SPCA using the local search algorithm and subgradient method, which are used as input values in MILP. The test data is in "pitdata.csv" giving a matrix of size 13 * 13. More datasets can be found in "datasets.zip".

## Approximation algorithms
For our paper "Exact and Approximation Algorithms for Sparse PCA", and detailed implementation of the algorithms and formulations, see: http://www.optimization-online.org/DB_FILE/2020/05/7802.pdf.

## Continuous relaxations
