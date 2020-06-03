#!/usr/bin/env python3
"""Analysis of shortest paths in cities
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import igraph
import matplotlib.collections as mc
import pickle
import pandas as pd

HOME = os.getenv('HOME')
sys.path.append(pjoin(HOME, 'projects/cityblocks/'))
import src.run_realcities as cityblocks



#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

##########################################################
if __name__ == "__main__":
    main()

