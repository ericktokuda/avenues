#!/usr/bin/env python

import argparse
import logging
import os, sys
from os.path import join as pjoin
from logging import debug, info
from itertools import product
from pathlib import Path
import socket
import time

import string
import igraph

import numpy as np
cimport numpy as np
# DTYPE = np.int
# ctypedef np.int_t DTYPE_t
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from libc.math cimport floor, sqrt

import pandas as pd
import copy
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
from subprocess import Popen, PIPE
from datetime import datetime
from multiprocessing import Pool
import pickle as pkl
# import torch

########################################################## Defines
def fast_random_choice(lst, probs, double randnum):
    return lst[np.searchsorted(probs.cumsum(), randnum)]

##########################################################
cpdef generate_waxman_adj(long n, long avgdegree, float alpha, float beta,
                          long xmin, long ymin, long xmax, long ymax):

    # cdef long maxnumvertices = n*avgdegree//2
    cdef int maxnumvertices = n*n//2
    cdef int[:, :] adj = np.ones((maxnumvertices, 2), dtype=np.intc)
    cdef double[:] x = np.zeros(n, dtype=np.double)
    cdef double[:] y = np.zeros(n, dtype=np.double)
    cdef int u, v, nodeid, i
    cdef double l

    for nodeid in range(n):
        x[nodeid] = xmin + ((xmax-xmin)*np.random.rand())
        y[nodeid] = ymin + ((ymax-ymin)*np.random.rand())

    l = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)

    i = 0
    for u in range(n):
        x1, y1 = x[u], y[u]
        for v in range(u + 1, n):
            x2, y2 = x[v], y[v]
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2)

            if np.random.rand() < beta * math.exp(-(d/l)*(1/alpha)):
                # adj[u, v] = 1 # just fill upper part of the matrix
                adj[i, 0] = u
                adj[i, 1] = v
                i += 1
    adj = adj[:i]
    return np.asarray(adj), np.asarray(x), np.asarray(y)

##########################################################
cpdef get_matrix_index_from_triu(int k, int n):
    cdef double i = n - 2 - floor(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    cdef double j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return int(i), int(j)

##########################################################
cpdef get_linear_index_from_triu(int i, int j, int n):
    return int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
