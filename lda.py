#!/usr/bin/env python

import numpy
import scipy
import scipy.weave as weave
import random
from datetime import datetime

seed = 12345

class LDA:
    def __init__(self, d_v_length, vocabs, T, alpha, beta, N=None, W=None, D=None):
        self.d_v_length = numpy.array(d_v_length)
        self.vocabs = vocabs
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.W = W
        self.D = D
        if not N:
            self.N = len(self.d_v_length)
            print self.N
        if not W:
            self.W = int(max(self.d_v_length[:,1])) + 1
        if not D:
            self.D = int(max(self.d_v_length[:,0])) + 1
        self.z = numpy.random.randint(0, high=self.T, size=self.N)

        with open("perplexity.embed.cpp") as f:
            self.perplexity_code = f.read()
        with open("train.embed.cpp") as f:
            self.train_code = f.read()

        self.initial_count()

    def initial_count(self):
        self.n_D_T =  numpy.zeros((self.D, self.T), dtype=int)
        self.n_W_T = numpy.zeros((self.W, self.T), dtype=int)
        self.n_D = numpy.zeros(self.D, dtype=int)
        self.n_T = numpy.zeros(self.T, dtype=int)
        d_v_length = self.d_v_length
        z = self.z
        n_D_T = self.n_D_T
        n_W_T = self.n_W_T
        n_D = self.n_D
        n_T = self.n_T
        N = self.N
        with open("initial_count.embed.cpp") as f:
            code = f.read()
        weave.inline(code, ['d_v_length', 'z', 'n_D_T', 'n_W_T', 'n_D', 'n_T', 'N'], type_converters=weave.converters.blitz)

    def train(self, validation=None):
        print "iter start"
        d = datetime.now()
        c = 0
        while True:
            c += 1
            perp = self.perplexity(validation)
            print "%d,%f" % (c,perp)
            if c == 150:
                break
            n_W_T = self.n_W_T
            n_D_T = self.n_D_T
            n_D = self.n_D
            n_T = self.n_T
            beta = self.beta
            alpha = self.alpha
            W = self.W
            T = self.T
            z = self.z
            N = self.N
            d_v_length = self.d_v_length

            weave.inline(self.train_code, ['n_W_T', 'n_D_T', 'n_D', 'n_T', 'beta', 'alpha', 'W', 'T', 'z', 'N', 'd_v_length'], type_converters=weave.converters.blitz, compiler="gcc")
        print "iter end"

    def perplexity(self, d_v_length=None):
        if d_v_length != None:
            N = len(d_v_length)
        else:
            d_v_length = self.d_v_length
            N = self.N

        phi_T_W = self.MAP_phi()
        theta_D_T = self.MAP_theta()
        T = self.T
        result = weave.inline(self.perplexity_code, ['d_v_length', 'phi_T_W', 'theta_D_T', 'T', 'N'], type_converters=weave.converters.blitz)
        return result

    def MAP_theta(self):
        _n_D = self.n_D[:, numpy.newaxis] # transposition
        return (self.n_D_T + self.alpha) / (_n_D + self.T * self.alpha)

    def MAP_phi(self):
        n_T_W = self.n_W_T.T
        _n_T = self.n_T[:, numpy.newaxis] # transposition
        return (n_T_W + self.beta) / (_n_T + self.W * self.beta)

if __name__ == "__main__":
    numpy.random.seed(seed)

    with open("vocab.nips.txt") as f:
        vocabularies = f.read().split("\n")
    with open("docword.nips.txt") as f:
        D = int(f.readline())
        W = int(f.readline())
        N = int(f.readline())
        lines = f.read().rstrip().split("\n")

    # lines = lines[:100434] # slicing for debugging

    print "init start"
    dec = lambda t: (t[0]-1, t[1]-1, t[2])
    d_v_length = [dec(map(int, line.split(" "))) for line in lines]

    # extracting for validation
    print "extracting for validation"
    N = len(d_v_length)
    indices = set(xrange(N))
    validation_indices = set(random.sample(indices, N / 10))
    d_v_length = numpy.array(d_v_length)
    d_v_l_validation = d_v_length[tuple(validation_indices),]
    d_v_length = d_v_length[tuple(indices - validation_indices),]
    if len(d_v_length) != N:
        # when slicing, D, W, and N may be wrong
        D = None
        W = None
        N = None

    print "instanciate LDA"
    lda = LDA(d_v_length, vocabularies, T=15, alpha=0.5, beta=0.5, N=N, W=W, D=D)
    lda.train(d_v_l_validation)

    print "=== phi_j_v ==="
    phi_T_W = lda.MAP_phi()
    topic_vocabs = []
    for j in range(lda.T):
        phi_j = phi_T_W[j]
        indices = phi_j.argsort()[::-1][:5]
        vocabs = ["%s:%.5f" % (vocabularies[i], phi_j[i]) for i in indices]
        topic_vocabs.append([vocabularies[i] for i in indices])
        print "  ".join(vocabs)

    print "=== theta_d_j ==="
    theta_D_T = lda.MAP_theta()
    for d in range(10):
        _d = lda.D / 10 * d
        theta_d = theta_D_T[_d]
        j = theta_d.argmax()
        print "%u: %.5f" % (_d, theta_d[j])
        print theta_d
        print topic_vocabs[j]

