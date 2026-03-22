#!/usr/bin/env python3
"""Compute Degree (D), Adjacency (A), and Laplacian (L) matrices
for an n-node star graph and verify against manual matrices for n=5.
"""

import numpy as np

def star_matrices(n, center=0):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        if i == center:
            continue
        A[center, i] = 1
        A[i, center] = 1
    D = np.diag(A.sum(axis=1))
    L = D - A
    return D, A, L

def manual_matrices_5():
    Dm = np.diag([4, 1, 1, 1, 1])
    Am = np.array([
        [0,1,1,1,1],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
    ], dtype=int)
    Lm = Dm - Am
    return Dm, Am, Lm

def main():
    D, A, L = star_matrices(5)
    Dm, Am, Lm = manual_matrices_5()

    print('Computed D:\n', D)
    print('Manual D:\n', Dm)
    print()
    print('Computed A:\n', A)
    print('Manual A:\n', Am)
    print()
    print('Computed L:\n', L)
    print('Manual L:\n', Lm)
    print()

    ok_D = np.array_equal(D, Dm)
    ok_A = np.array_equal(A, Am)
    ok_L = np.array_equal(L, Lm)
    print('Matches manual? D:', ok_D, ' A:', ok_A, ' L:', ok_L)

    print('L is symmetric:', np.allclose(L, L.T))
    print('Each row of L sums to zero:', np.allclose(L.sum(axis=1), 0))

if __name__ == '__main__':
    main()