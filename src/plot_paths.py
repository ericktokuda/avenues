#!/usr/bin/env python3
"""one-line docstring
"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from myutils import info, create_readme
import pandas as pd

##########################################################
def plot_global(df, outdir):
    """Plot global avg path len """
    info(inspect.stack()[0][3])
    nrows = 1;  ncols = 1; figscale = 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*figscale, nrows*figscale))

    y = df.g_pathlen_mean
    axs.errorbar(list(range(len(y))), y, yerr=df.g_pathlen_std)
    axs.set_ylim(0, np.max(y)*1.2)
    plt.savefig(pjoin(outdir, 'global.png'))

##########################################################
def plot_local_individually(df, outdir):
    """Plot each line individually """
    info(inspect.stack()[0][3] + '()')

    nrows = 1;  ncols = 1; figscale = 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*figscale, nrows*figscale))

    cols = []
    for col in df.columns:
        if not col.startswith('pathlen_'): continue
        y = df[col]
        axs.plot(list(range(len(df))), y, label=col)
        cols.append(col)

    axs.set_ylabel('Length')
    axs.set_title('Individual path length for {} regions'.format(len(cols)))
    plt.savefig(pjoin(outdir, 'local_individually.png'))

##########################################################
def plot_local_mean(df, outdir):
    """Plot each line individually """
    info(inspect.stack()[0][3] + '()')

    cols = []
    for col in df.columns:
        if not col.startswith('pathlen_'): continue
        cols.append(col)

    vals = df[cols].values
    
    nrows = 1;  ncols = 1; figscale = 8
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*figscale, nrows*figscale))

    axs.errorbar(list(range(len(df))), np.mean(vals, axis=1),
            yerr=np.std(vals, axis=1))
    axs.set_ylabel('Length')
    axs.set_title('Average path lengths (across {} regions)'.format(len(cols)))
    plt.savefig(pjoin(outdir, 'local_mean.png'))

##########################################################
def plot_corr_all(df, localfeats, outdir):
    """Plot correlation between independent variables """
    info(inspect.stack()[0][3] + '()')

    # data = dict()
    data = pd.DataFrame()

    data['g_nbridges'] = df.nbridges
    data['g_naccess'] = df.naccess
    data['g_pathlen_mean'] = df.g_pathlen_mean
    data['g_pathlen_std'] = df.g_pathlen_std
    # data['g_betwmean'] = df.g_betwmean
    # data['g_betwstd'] = df.g_betwstd

    for feat in localfeats:
        cols = []
        for col in df.columns:
            if col.startswith(feat + '_'): cols.append(col)

        data['l_' + feat + '_mean'] = np.mean(df[cols].values, axis=1)
        data['l_' + feat + '_std'] = np.std(df[cols].values, axis=1)
    
    corr = data.corr()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.savefig(pjoin(outdir, 'heatmap.png'))

##########################################################
def plot_hists(df, localfeats, outdir):
    """Plot histogram of the features"""
    info(inspect.stack()[0][3] + '()')
    m = len(df)
    for col1 in localfeats[1:]:
        x = np.zeros(m, dtype=float)
        for k in range(30):
            x += df['{}_{:03d}'.format(col1, k)].values

        nrows = 1;  ncols = 1; figscale = 8
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(ncols*figscale, nrows*figscale))
        axs.hist(x)
        axs.set_xlabel(col1)
        plt.tight_layout()
        plt.savefig(pjoin(outdir, 'hist_{}.png'.format(col1)))
        plt.close()

def plot_pairwise_points(df, localfeats, outdir):
    m = len(df)
    for col1 in localfeats[1:]:
        x = np.zeros(m, dtype=float)
        for k in range(30):
            x += df['{}_{:03d}'.format(col1, k)].values
        for col2 in ['pathlen']:
            y = np.zeros(m, dtype=float)
            for k in range(30):
                y += df['{}_{:03d}'.format(col2, k)].values

        nrows = 1;  ncols = 1; figscale = 8
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(ncols*figscale, nrows*figscale))
        axs.scatter(x, y)
        axs.set_xlabel(col1)
        axs.set_ylabel('Pathlen')
        plt.tight_layout()
        plt.savefig(pjoin(args.outdir, '{}_{}.png'.format(col1, col2)))
        plt.close()

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='/tmp/out/results.csv', help='Path to the results.csv file')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)
    readmepath = create_readme(sys.argv, args.outdir)

    df = pd.read_csv(args.results)
    localfeats = ['pathlen', 'degree', 'betwv', 'divers', 'clucoeff', 'closeness']

    plot_global(df, args.outdir)
    plot_local_individually(df, args.outdir)
    plot_local_mean(df, args.outdir)
    plot_corr_all(df, localfeats, args.outdir)
    plot_hists(df, localfeats, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
