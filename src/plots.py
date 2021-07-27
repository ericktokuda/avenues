#!/usr/bin/env python3
"""Plots
"""

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from myutils import info, create_readme

##########################################################
def plot_means_maxs(allresdir, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    meanmax = get_means_maxs(allresdir)

    labels = ['mean', 'max']
    W = 640; H = 480
    for i in range(2):
        m = meanmax[i]; label = labels[i]
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        ax.scatter(m.keys(), m.values())
        ax.tick_params(axis='x', labelrotation=90)
        plt.tight_layout()
        outpath = pjoin(outdir, '{}.png'.format(label))
        plt.savefig(outpath)

##########################################################
def get_means_maxs(allresdir):
    """Get mean and max for each subdir corresponding to a city"""
    info(inspect.stack()[0][3] + '()')
    speed = '1.0'
    means = {}; maxs = {}
    for d in sorted(os.listdir(allresdir)):
        f = pjoin(allresdir, d, 'results.csv')
        if not os.path.exists(f): continue
        city, field2, field3 = d.split('_')
        brspacing = field2.split('brspa')[1]
        brspeed = field3.split('brspe')[1]
        if not brspeed == speed: continue
        df = pd.read_csv(f)
        if city in means.keys():
            info('Overwriting means[{}]'.format(city))
        means[city], maxs[city] = df.gain.mean(), df.gain.max()

    return means, maxs

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allresdir', required=True, help='Dir containing the results for each city')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    plot_means_maxs(args.allresdir, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
