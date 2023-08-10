# Exact and Approximation Algorithms for Sparse PCA (SPCA)

This project contains the implementations of exact algorithms, approximation algorithms, and continuous relaxations for SPCA in our paper "Exact and Approximation Algorithms for Sparse PCA": http://www.optimization-online.org/DB_FILE/2020/05/7802.pdf.

## Exact algorithms
We propose three equivalent mixed-integer convex programs to solve SPCA to optimality, denoted by MISDP (6), MISDP (15), and MILP (22) in our paper. The implementations of these exact formulations can be found in the "exact algorithms" directory.


To be specific, we design a branch-and-cut algorithm with the closed-form cuts to solve MISDP (6), and the implementation is available in the "Branchandcut.py" file.\
We also customize a branch-and-cut algorithm to solve MISDP (15), and the implementation is available in the "Branchandcut_15.py" file.\
For MILP (22),  the implementation is available in the "MILP.py" file.\
Lastly, we use the solver Gurobi 9.5.2 to directly solve the SPCA, and the implementation is available in the "spca_gurobi.py" file.


## Approximation algorithms
We investigate two approximation algorithms for solving SPCA: greedy and local search algorithms, and compare them with the existing truncation algorithm, randomized algorithm, and SDP-based algorithm. The implementations can be found in the "approximation algorithms" directory


To be specific, The implementations of truncation, greedy, and local search algorithms can be found in the "lower_bounds.py" file.\
The implementation of randomized algorithm is available in the "rspca.py" file.\
The implementation of SDP-based algorithm is available in the "spca_sdp.py" file.

## Continuous relaxations

## Datasets
