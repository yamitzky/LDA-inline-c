#!/usr/bin/env python

import numpy
import scipy
import scipy.weave as weave
from random import random
from datetime import datetime

beta = 0.5
alpha = 0.5
seed = 12345

if __name__ == "__main__":
    numpy.random.seed(seed)

    with open("vocab.nips.txt") as f:
        vocabularies = f.read().split("\n")
    with open("docword.nips.txt") as f:
        D = int(f.readline())
        W = int(f.readline())
        n = int(f.readline())
        lines = f.read().split("\n") # TODO: delete slicing

    lines = lines[:-1]
    W = max([int(line.split(" ")[1]) for line in lines])
    D = int(lines[-1].split(" ")[0])
    n = len(lines)

    T = 15
    z = numpy.random.randint(0, high=T, size=n)
    n_d_j =  numpy.zeros((D, T), dtype=int)
    n_v_j = numpy.zeros((W, T), dtype=int)
    d_v_length = numpy.zeros((n, 3), dtype=int)
    n_d = numpy.zeros(D, dtype=int)
    n_j = numpy.zeros(T, dtype=int)

    print "init start"
    dec = lambda t: (t[0]-1, t[1]-1, t[2])
    d_v_length = [dec(map(int, line.split(" "))) for line in lines]
    d_v_length = numpy.array(d_v_length)
    code = """
    for(int i = 0; i < n; i++) {
        int d = d_v_length(i,0);
        int v = d_v_length(i,1);
        int length = d_v_length(i,2);
        int j = z(i);
        n_d_j(d,j) += length;
        n_v_j(v,j) += length;
        n_d(d) += length;
        n_j(j) += length;
    }
    """
    weave.inline(code, ['d_v_length', 'z', 'n_d_j', 'n_v_j', 'n_d', 'n_j', 'n'], type_converters=weave.converters.blitz)

    print "iter start"
    d = datetime.now()
    c = 0
    while True:
        c += 1
        if c % 10 == 0:
            print c
            print datetime.now() - d
            print "-----"
        if c % 1000 == 0:
            break
        code = """
        for (int i = 0; i < n; i++) {
            float Q_i[T];
            int z_i = z(i);
            int d = d_v_length(i, 0);
            int v = d_v_length(i, 1);
            int length = d_v_length(i, 2);
            for(int j = 0; j < T; j++) {
                int n_v_j_i = n_v_j(v,j);
                int n_d_j_i = n_d_j(d,j);
                int n_j_i = n_j(j);
                int n_d_i = n_d(d) - length;
                if (z_i == j) {
                  n_v_j_i -= length;
                  n_d_j_i -= length;
                  n_j_i -= length;
                }
                double q_i = (n_v_j_i + beta) * (n_d_j_i + alpha) /
                        (n_j_i + W * beta) / (n_d_i + T * alpha);
                double Q_i_sum = (j > 0)? Q_i[j-1] : 0;
                Q_i[j] = Q_i_sum + q_i;
            }
            float u = random() / (float)RAND_MAX;
            int j;
            for (j = 0; j < T; j++) {
                if(Q_i[j] / Q_i[T-1] >= u) break;
            }
            if (z_i != j) {
                z(i) = j;
                n_d_j(d,z_i) -= length;
                n_v_j(v,z_i) -= length;
                n_j(z_i) -= length;
                n_d_j(d,j) += length;
                n_v_j(v,j) += length;
                n_j(j) += length;
            }
        }
        """
        weave.inline(code, ['n_v_j', 'n_d_j', 'n_j', 'n_d', 'beta', 'alpha', 'W', 'T', 'z', 'n', 'd_v_length'], type_converters=weave.converters.blitz, compiler="gcc")


    phi_j_v = numpy.array((T,W))
    phi_j_v[j,v] = (n_v_j[v,j] + beta) / (n_j[j] + W * beta)
    for j in range(T):
        phi_j_sorted = sorted(phi_j)
        phi_j_sorted = phi_j_sorted[-5:]
        indexies = [phi_j.index(phi) for phi in phi_j_sorted]
        vs = [vocabularies[i] for i in indexies]
        print vs
    print "======="

