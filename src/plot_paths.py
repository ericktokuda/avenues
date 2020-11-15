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
# import matplotlib; matplotlib.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
from myutils import info, create_readme
import pandas as pd
import scipy.stats

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
    # data['g_pathlen_std'] = df.g_pathlen_std
    data['g_betwv_mean'] = df.g_betwv_mean
    # data['g_betwv_std'] = df.g_betwstd
    data['g_assort_mean'] = df.g_assort_mean
    # data['g_assort_std'] = df.g_assort_std
    data['g_clucoeff_mean'] = df.g_clucoeff_mean
    # data['g_clucoeff_std'] = df.g_clucoeff_std
    data['g_divers_mean'] = df.g_divers_mean
    # data['g_divers_std'] = df.g_divers_std
    data['g_clos_mean'] = df.g_clos_mean
    # data['g_clos_std'] = df.g_clos_std

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

##########################################################
def plot_densities(df, localfeats, outdir):
    """Plot histogram of the features"""
    info(inspect.stack()[0][3] + '()')
    m = len(df)

    for col1 in localfeats[1:]:
        x = np.zeros(m, dtype=float)
        for k in range(100):
            x += df['{}_{:03d}'.format(col1, k)].values

        # kde = scipy.stats.gaussian_kde(x, bw_method='scott')
        kde = scipy.stats.gaussian_kde(x, bw_method=.346)

        xtest = np.linspace(np.min(x), np.max(x), 100)
        ytest = kde.evaluate(xtest)

        nrows = 1;  ncols = 1; figscale = 8
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(ncols*figscale, nrows*figscale))
        axs.hist(x, density=True)
        axs.plot(xtest, ytest)
        axs.set_xlabel(col1)
        plt.tight_layout()
        plt.savefig(pjoin(outdir, 'dens_{}.png'.format(col1)))
        plt.close()

##########################################################
def compare_densities(df, localfeats, outdir):
    """Plot histogram of the features"""
    info(inspect.stack()[0][3] + '()')
    m = len(df)
    n = 100 # num realizations

    for col1 in localfeats[1:]:
        nrows = 1;  ncols = 1; figscale = 8
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(ncols*figscale, nrows*figscale))

        # Before bridge
        x = df['{}_{:03d}'.format(col1, 0)].values
        # kde = scipy.stats.gaussian_kde(x, bw_method='scott')
        kde = scipy.stats.gaussian_kde(x, bw_method=.346)
        xtest = np.linspace(np.min(x), np.max(x), 100)
        ytest = kde.evaluate(xtest)
        # axs.hist(x, density=True)
        axs.plot(xtest, ytest, label='Before')

        # After bridge
        x = np.zeros(m, dtype=float)
        for k in range(1, n):
            x += df['{}_{:03d}'.format(col1, k)].values
        x /=  (n-1) # average
        # kde = scipy.stats.gaussian_kde(x, bw_method='scott')
        kde = scipy.stats.gaussian_kde(x, bw_method=.346)
        xtest = np.linspace(np.min(x), np.max(x), 100)
        ytest = kde.evaluate(xtest)
        # axs.hist(x, density=True)
        axs.plot(xtest, ytest, label='After')

        axs.set_xlabel(col1)
        fig.legend()
        plt.tight_layout()
        plt.savefig(pjoin(outdir, 'dens_comparison_{}.png'.format(col1)))
        plt.close()

##########################################################
def plot_densities_all(localfeats, outdir):
    """Plot histogram of the features"""
    info(inspect.stack()[0][3] + '()')
    dfs = {}
    sp = '1.00'
    dfs['bar'] = pd.read_csv('/home/frodo/results/bridges/20201026-4cities/bar_s1000_n200_sp{}/results.csv'.format(sp))
    dfs['dub'] = pd.read_csv('/home/frodo/results/bridges/20201026-4cities/dub_s1000_n200_sp{}/results.csv'.format(sp))
    dfs['man'] = pd.read_csv('/home/frodo/results/bridges/20201026-4cities/man_s1000_n200_sp{}/results.csv'.format(sp))
    dfs['par'] = pd.read_csv('/home/frodo/results/bridges/20201026-4cities/par_s1000_n200_sp{}/results.csv'.format(sp))

    m = len(dfs['bar'])

    for col1 in localfeats[1:]:
        x = np.zeros(m, dtype=float)
        nrows = 1;  ncols = 1; figscale = 8
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(ncols*figscale, nrows*figscale))

        for kk in dfs.keys():
            df = dfs[kk]
            for k in range(100):
                x += df['{}_{:03d}'.format(col1, k)].values

            # kde = scipy.stats.gaussian_kde(x, bw_method='scott')
            kde = scipy.stats.gaussian_kde(x, bw_method=.346)

            xtest = np.linspace(np.min(x), np.max(x), 100)
            ytest = kde.evaluate(xtest)

            # axs.hist(x, density=True)
            axs.plot(xtest, ytest, label=kk)
        axs.set_xlabel(col1)
        fig.legend()
        plt.tight_layout()
        plt.savefig(pjoin(outdir, 'all_dens_{}.png'.format(col1)))
        plt.close()

##########################################################
def plot_pairwise_points(df, localfeats, outdir):
    m = len(df)

    # y = []
    pathlenorig = []
    pathlennew = []
    col2 = 'pathlen'
    for col in sorted(df.columns):
        if not col.startswith(col2): continue
        # y.append(np.mean(df[col]))
        pathlennew.append(np.mean(df[col].loc[1:]))
        pathlenorig.append(df[col].loc[0])

    y = np.array(pathlennew) - np.array(pathlenorig)

    for col1 in localfeats[1:]:
        nrows = 1;  ncols = 1; figscale = 8
        fig, axs = plt.subplots(nrows, ncols,
                    figsize=(ncols*figscale, nrows*figscale))

        x = []
        for col in df.columns:
            if not col.startswith(col1): continue
            # x.append(np.mean(df[col]))
            x.append(df[col].loc[0])

        epsilon = 0.0
        inds = np.where(np.abs(y) > epsilon)[0]
        x = np.array(x)[inds]
        y = np.array(y)[inds]

        corr, _ = scipy.stats.pearsonr(x, y)
        axs.scatter(x, y, alpha=0.3)
        axs.set_xlabel(col1)
        axs.set_ylabel('Delta pathlen')
        axs.set_title('Pearson:{:.02f}, n:{}'.format(corr, len(inds)))
        plt.tight_layout()
        plt.savefig(pjoin(outdir, 'pair_{}_{}.png'.format(col1, col2)))
        plt.close()

##########################################################
def get_correlation_table(dfs, localfeats):
    corrs = np.zeros((len(localfeats)-1, len(dfs)), dtype=float)
    for i, l in enumerate(localfeats[1:]):
        for j, df in enumerate(dfs):
            # get y axis of the correlation
            pathlenorig = []
            pathlennew = []
            col2 = 'pathlen'
            for col in sorted(df.columns):
                if not col.startswith(col2): continue
                # y.append(np.mean(df[col]))
                pathlennew.append(np.mean(df[col].loc[1:]))
                pathlenorig.append(df[col].loc[0])

            y = np.array(pathlennew) - np.array(pathlenorig)

            # get x axis
            x = []
            for col in df.columns:
                if not col.startswith(l): continue
                # x.append(np.mean(df[col]))
                x.append(df[col].loc[0])

            # filter out null values
            epsilon = 0.0
            inds = np.where(np.abs(y) > epsilon)[0]

            x = np.array(x)[inds]
            y = np.array(y)[inds]

            corr, _ = scipy.stats.pearsonr(x, y)

            print(i, j, len(x), corr)
            corrs[i, j] = corr

    return corrs

##########################################################
def plot_heatmap(localfeats, outdir):
    # templ = '/home/frodo/results/bridges/20201016-4cities/C_s1000_n200/results.csv'
    templ = '/home/frodo/results/bridges/20201026-4cities/C_s1000_n200_sp0.50/results.csv'
    cities = ['bar', 'dub', 'man', 'par']
    mypaths = {c: templ.replace('C', c) for c in cities}
    dfs = [pd.read_csv(p) for p in mypaths.values()]

    # nmeasures x ncities
    corrs = get_correlation_table(dfs, localfeats)

    fig, ax = plt.subplots()
    im = ax.imshow(corrs, cmap='PiYG')

    # We want to show all ticks...
    ax.set_xticks(np.arange(corrs.shape[1]))
    ax.set_yticks(np.arange(corrs.shape[0]))
    ax.set_xticklabels(cities)
    ax.set_yticklabels(localfeats[1:])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(corrs.shape[0]):
        for j in range(corrs.shape[1]):
            text = ax.text(j, i, '{:.02f}'.format(corrs[i, j]),
                           ha="center", va="center", color="k")

    ax.set_title("Correlation with gain in the avg. short. path length")
    fig.tight_layout()

    outpath = pjoin(outdir, 'corr_heatmap.png')
    plt.savefig(outpath)

##########################################################
def plot_avg_path_lengths(localfeats, outdir):
    # templ = '/home/frodo/results/bridges/20201026-4cities/C_s1000_n200_spS/results.csv'
    templ = '/home/dufresne/temp/20201113-bridges/C_len0.5_spS/results.csv'
    # cities = ['barcelona', 'dublin', 'manchester', 'paris']
    cities = ['barcelona']
              # 'wx0.001', 'wx0.005', 'wx0.010']
    # speeds = ['0.50', '1.00', '2.00']
    speeds = ['0.25', '0.5', '0.75', '1.0', '1.5', '2.0', '4.0']

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    outpath = pjoin(outdir, 'pathlen_speeds.png')
    col = 'g_pathlen_mean'

    for c in cities:
        dfs = []

        avgpathlens = []
        stdpathlens = []
        for s in speeds:
            df = pd.read_csv(templ.replace('C', c).replace('S', s))
            pathlens = df[col].loc[1:] # Idx 0 is without bridges
            # pathlens /= df[col].loc[0]
            avgpathlens.append(np.mean(pathlens))
            stdpathlens.append(np.std(pathlens))

        ax.errorbar([float(k) for k in speeds], avgpathlens, yerr=stdpathlens,
                    label=c)

    ax.set_xlabel('Bridge speed')
    ax.set_ylabel('Relative average path length')
    fig.legend()
    # plt.show()
    plt.savefig(outpath)

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='/tmp/out/results.csv', help='Path to the results.csv file')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # df = pd.read_csv(args.results)
    localfeats = ['pathlen', 'degree', 'betwv', 'divers', 'clucoeff', 'clos']

    # plot_global(df, args.outdir)
    # plot_local_individually(df, args.outdir)
    # plot_local_mean(df, args.outdir)
    # plot_corr_all(df, localfeats, args.outdir)
    # plot_hists(df, localfeats, args.outdir)
    # plot_densities(df, localfeats, args.outdir)
    # compare_densities(df, localfeats, args.outdir)
    # plot_densities_all(localfeats, args.outdir)
    # plot_heatmap(localfeats, args.outdir )
    # plot_pairwise_points(df, localfeats, args.outdir )
    plot_avg_path_lengths(localfeats, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
