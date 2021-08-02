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
import matplotlib.collections as mc
import pandas as pd
from myutils import info, create_readme, graph
from scipy.stats import pearsonr
import pickle

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
    plt.close()

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
    plt.close()

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

    # ax.set_ylim(0.988, 1.002)
    ax.set_title("Correlation with gain in the avg. short. path length")
    fig.tight_layout()

    outpath = pjoin(outdir, 'corr_heatmap.png')
    plt.savefig(outpath)

##########################################################
def plot_avg_path_lengths(localfeats, outdir):
    # templ = '/home/frodo/results/bridges/20201026-4cities/C_s1000_n200_spS/results.csv'
    # templ = '/home/frodo/results/bridges/20210119-avenues/C_len2.0_spS/results.csv'
    templ = '/home/frodo/results/bridges/20210412-bridges/C_brlenBL_ndetoursND_spS/results.csv'
    # templ = '/tmp/C_len2.0_spS/results.csv'
    # cities = ['barcelona', 'dublin', 'manchester', 'paris']
    cities = ['barcelona', 'dublin', 'paris']
    # cities = ['barcelona']
              # 'wx0.001', 'wx0.005', 'wx0.010']
    # speeds = ['0.50', '1.00', '2.00']
    speeds = ['0.25', '1.0', '4.0']

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
            print(templ.replace('C', c).replace('S', s))
            pathlens = df[col].loc[1:] # Idx 0 is without bridges
            # pathlens /= df[col].loc[0]
            pathlens /= np.max(pathlens)
            avgpathlens.append(np.mean(pathlens))
            stdpathlens.append(np.std(pathlens))

        ax.errorbar([float(k) for k in speeds], avgpathlens, yerr=stdpathlens,
                    label=c)

    # ax.set_ylim(0.988, 1.002)
    ax.set_xlabel('Bridge speed')
    ax.set_ylabel('Relative average path length')
    fig.legend()
    # plt.show()
    plt.savefig(outpath)

##########################################################
def get_bridge_coords(resdir, ndigits):
    brcoordsexact = pickle.load(
        open(pjoin(resdir, 'brcoordsexact.pkl'), 'rb'))
    brcoords = pickle.load(
        open(pjoin(resdir, 'brcoords.pkl'), 'rb'))
    avenues = pickle.load(
        open(pjoin(resdir, 'avenues.pkl'), 'rb'))

    data = np.zeros((len(brcoords), 10), dtype=float)
    for i in range(brcoords.shape[0]):
        data[i, :4] = brcoordsexact[i, :]
        delta = brcoordsexact[i, :2] - brcoordsexact[i, 2:]
        angrad = math.atan2(delta[1], delta[0])
        data[i, 4] = angrad

        data[i, 5:9] = brcoords[i, :]
        delta = brcoords[i, :2] - brcoords[i, 2:]
        angrad = math.atan2(delta[1], delta[0])
        data[i, 9] = angrad

    return np.around(data, ndigits)

##########################################################
def load_all_results(resultsdir, outdir):
    """Concatenate all results into a single tabular file."""
    csvpath = pjoin(outdir, 'resultsall.csv')
    info(inspect.stack()[0][3] + '()')

    if os.path.exists(csvpath):
        info('Loading existing {}'.format(csvpath))
        return pd.read_csv(csvpath)

    dirs = os.listdir(resultsdir)

    df = []
    ndigits = 4
    rows = np.zeros((0, 13), dtype=object)
    for d in sorted(dirs):
        respath = pjoin(resultsdir, d, 'results.csv')
        if not os.path.exists(respath): continue
        df2 = pd.read_csv(respath)
        if len(df) == 0: df = df2
        else: df = pd.concat([df, df2], ignore_index=True)

    df.to_csv(csvpath, index=False)
    return df

##########################################################
def plot_3d_cmap(dfall, exact, outdirarg):
    suff = 'exact' if exact else ''
    outdir = pjoin(outdirarg, '3d_cmap_' + suff)
    os.makedirs(outdir, exist_ok=True)
    import plotly.express as px

    dfall.brangle = dfall.brangle *180 / np.pi
    dfall.brexactangle = dfall.brexactangle *180 / np.pi

    x = 'brexactsrcx' if exact else 'brsrcx'
    y = 'brexactsrcy' if exact else 'brsrcy'
    z = 'brexactangle' if exact else 'brangle'
    for city in np.unique(dfall.city):
        for sp in np.unique(dfall.brspeed):
            df = dfall.loc[(dfall.city == city) & (dfall.brspeed == sp)]

            gains = np.array(df.gain)
            threshid  = int(.97 * len(df)) # 10% wiht the greatest values
            thresh = sorted(gains)[threshid]

            texts = []
            for j in range(len(df)):
                texts.append(str(j) if gains[j] > thresh else '')

            fig = px.scatter_3d(df, x=x, y=y, z=z,
                                # text=texts,
                                color='gain',
                                color_continuous_scale=px.colors.sequential.Viridis,
                                opacity=.7,
                                title='Bridge gain for points in the grid')

            fig.update_scenes(xaxis_title='x')
            fig.update_scenes(yaxis_title='y')
            fig.update_scenes(zaxis_title='Angle')
            outpath = pjoin(outdir, '{}_sp{}_{}_3dcmap.html'.format(city, sp, suff))
            fig.write_html(outpath)

##########################################################
def plot_3d_sizes(dfall, exact, outdirarg):
    outdir = pjoin(outdirarg, '3d_sizes')
    os.makedirs(outdir, exist_ok=True)
    x = 'brexactsrcx' if exact else 'brsrcx'
    y = 'brexactsrcy' if exact else 'brsrcy'
    z = 'brexactangle' if exact else 'brangle'
    suff = 'exact' if exact else ''

    for city in np.unique(dfall.city):
        for sp in np.unique(dfall.brspeed):
            df = dfall.loc[(dfall.city == city) & (dfall.brspeed == sp)]

            import plotly.graph_objects as go
            gain = df.gain.to_numpy()
            gain = np.log(gain+.0000001)
            normgain = ((gain - np.min(gain)) / (np.max(gain) - np.min(gain)))
            # normgain += 1.01

            fig = go.Figure()
            for i in range(len(df)):
                d = df.iloc[i]
                fig.add_trace(go.Scatter3d(x=[d[x]], y=[d[y]],
                                           z=[d[z]],
                                           showlegend=False,
                                           marker=dict(
                                               size=normgain[i]*10,
                                               color='red',
                                               opacity=.5)
                                           )
                              )


            # outpath = pjoin(outdir, '{}_sp{}.html'.format(city, sp))
            outpath = pjoin(outdir, '{}_sp{}_{}_3dsize.html'.format(city, sp, suff))
            fig.write_html(outpath)

##########################################################
def load_list_of_lists(fpath):
    fh = open(fpath)
    mylist = []

    for l in fh:
        mylist.append([int(ll) for ll in l.strip().split(',')])

    fh.close()
    return mylist

##########################################################
def plot_grid(resdir, graphmldir, outdirarg):
    """Plot all avenues coloured by gain """
    info(inspect.stack()[0][3] + '()')

    outdir = pjoin(outdirarg, 'grid')
    os.makedirs(outdir, exist_ok=True)
    plotted = set()
    for d in sorted(os.listdir(resdir)):
        info('d:{}'.format(d))
        dpath = pjoin(resdir, d)
        if not os.path.isdir(dpath): continue
        df = pd.read_csv(pjoin(dpath, 'results.csv'))
        city = df.loc[0].city
        speed = df.loc[0].brspeed
        gains = df.gain.to_numpy()

        if city in plotted: continue
        plotted.add(city)

        gridpath = pjoin(resdir, d, 'brcoordsexact.pkl')
        # avs = load_list_of_lists(avpath)
        gridedges = pickle.load(open(gridpath, 'rb'))
        # grid0, grid1 = gridedges[:, 0:2], gridedges[:, 2:]

        g = graph.simplify_graphml(pjoin(graphmldir, city + '.graphml'))
        coords = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
        # coords[avs[3]] gives the coordinates of the av points

        shppath = ''
        ax = None
        from myutils import plot

        for attrs in [('lon', 'lat'), ('posx', 'posy'), ('x', 'y')]:
            if attrs[0] in g.vertex_attributes():
                xattr = 'x'; yattr = 'y'

        vcoords = np.array([(x, y) for x, y in zip(g.vs[xattr], g.vs[yattr])])
        vcoords = vcoords.astype(float)

        ecoords = []
        for e in g.es:
            ecoords.append([ [float(g.vs[e.source]['x']), float(g.vs[e.source]['y'])],
                    [float(g.vs[e.target]['x']), float(g.vs[e.target]['y'])], ])

        ax = plot.plot_graph_coords(vcoords, ecoords, ax, shppath)
        ax.scatter(gridedges[:, 0], gridedges[:, 1], c='red')

        # Plot avenues with width proportional to the gain
        segs = []
        for i in range(gridedges.shape[0]):
            src, tgt = gridedges[i, 0:2], gridedges[i, 2:]
            edge = [src, tgt]
            segs.append([src, tgt])

        segs = mc.LineCollection(segs, colors='b',
                                 linewidths=1, alpha=.4) # edges
        ax.add_collection(segs)

        ax.axis('off')
        plt.tight_layout()
        plotpath = pjoin(outdir, '{}_grid.png'.format(city))
        plt.savefig(plotpath)
        plt.close()

##########################################################
def plot_avenues_all(resdir, graphmldir, outdirarg):
    """Plot all avenues coloured by gain """
    info(inspect.stack()[0][3] + '()')
    outdir = pjoin(outdirarg, 'gains')
    os.makedirs(outdir, exist_ok=True)

    for d in sorted(os.listdir(resdir)):
        info('d:{}'.format(d))
        dpath = pjoin(resdir, d)
        if not os.path.isdir(dpath): continue
        df = pd.read_csv(pjoin(dpath, 'results.csv'))
        city = df.loc[0].city
        speed = df.loc[0].brspeed
        gains = df.gain.to_numpy()

        avpath = pjoin(resdir, d, 'avenues.txt')
        avs = load_list_of_lists(avpath)

        g = graph.simplify_graphml(pjoin(graphmldir, city + '.graphml'))
        coords = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
        # coords[avs[3]] gives the coordinates of the av points

        shppath = ''
        ax = None
        from myutils import plot

        for attrs in [('lon', 'lat'), ('posx', 'posy'), ('x', 'y')]:
            if attrs[0] in g.vertex_attributes():
                xattr = 'x'; yattr = 'y'

        vcoords = np.array([(x, y) for x, y in zip(g.vs[xattr], g.vs[yattr])])
        vcoords = vcoords.astype(float)

        ecoords = []
        for e in g.es:
            ecoords.append([ [float(g.vs[e.source]['x']), float(g.vs[e.source]['y'])],
                    [float(g.vs[e.target]['x']), float(g.vs[e.target]['y'])], ])

        ax = plot.plot_graph_coords(vcoords, ecoords, ax, shppath)

        # Plot avenues with width proportional to the gain
        threshid  = int(.97 * len(df)) # 10% wiht the greatest values
        thresh = sorted(gains)[threshid]
        refgain = 5 / np.max(gains)
        avcoords = []
        ws = []
        acc = 0
        cmap = matplotlib.cm.get_cmap('viridis')
        normgains = (gains  - np.min(gains))/  (np.max(gains) - np.min(gains))
        colors = []
        for i, av in enumerate(avs):
            src = av[0]
            gain = gains[i] * refgain
            ws.extend([gain] * (len(av) - 1))
            for tgt in av[1:]:
                avcoords.append(coords[[src, tgt]])
                src = tgt

            if gains[i] < thresh: # skip if below gain
                continue

            acc += 1
            colors.append(cmap(normgains[i]))
            if gains[i] > thresh: # Plot label just if above threshold
                midid = int(len(av) / 2)
                textcoords = (coords[av[midid-1]] + coords[av[midid]]) / 2
                # ax.text(textcoords[0], textcoords[1], i)

        info('Avenues plotted :{}'.format(acc))

        
        # segs = mc.LineCollection(avcoords, colors='r',
        segs = mc.LineCollection(avcoords, colors=colors,
                                 linewidths=ws, alpha=.8) # edges
        ax.add_collection(segs)
        plt.colorbar(segs)
        ax.axis('off')
        plt.tight_layout()
        plotpath = pjoin(outdir, '{}_sp{}_gains.png'.format(city, speed))
        plt.savefig(plotpath)
        plt.close()

##########################################################
def plot_means_maxs(means, maxs, outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')
    data = [means, maxs]
    labels = ['mean', 'max']
    cols = [ 'gain' + l for l in labels ]
    W = 640; H = 480
    for i in range(2):
        m = data[i]; label = labels[i]
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        ax.scatter(m.city, m[cols[i]])
        ax.tick_params(axis='x', labelrotation=90)
        plt.tight_layout()
        outpath = pjoin(outdir, '{}.png'.format(label))
        plt.savefig(outpath)
        plt.close()

##########################################################
def get_means_maxs(allresdir, outdir):
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

    means2 = []; maxs2 = []
    for c in means.keys():
        means2.append([c, means[c]] )
        maxs2.append([c, maxs[c]] )
    meansdf = pd.DataFrame(means2, columns=['city', 'gainmean'])
    meansdf.to_csv(pjoin(outdir, 'means.csv'), index=False)
    maxsdf = pd.DataFrame(maxs2, columns=['city', 'gainmax'])
    maxsdf.to_csv(pjoin(outdir, 'maxs.csv'), index=False)
    return meansdf, maxsdf

##########################################################
def plot_correlation(means, featscsv, outdir):
    feats = pd.read_csv(featscsv)
    featlabels = list(feats.columns)
    featlabels.remove('city')

    for l in featlabels:
        W = 640; H = 480
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        ax.scatter(means['gainmean'], feats[l])
        p = pearsonr(means['gainmean'], feats[l])[0]
        ax.set_title('Mean gain X {} ({:.02f})'.format(l, p))
        outpath = pjoin(outdir, l)
        plt.savefig(outpath)
        plt.close()

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphmldir', required=True, help='Dir containing the graphml')
    parser.add_argument('--allresdir', required=True, help='Dir containing the results for each city')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    means, maxs = get_means_maxs(args.allresdir, args.outdir)
    plot_means_maxs(means, maxs, args.outdir)
    featscsv = pjoin(args.allresdir, 'features.csv')
    plot_correlation(means, featscsv, args.outdir)

    dfall = load_all_results(args.allresdir, args.outdir)
    plot_3d_cmap(dfall.copy(), True, args.outdir)
    plot_3d_cmap(dfall.copy(), False, args.outdir)
    plot_3d_cmap(dfall.copy(), True, args.outdir)
    plot_3d_sizes(dfall.copy(), True, args.outdir)

    pardir = os.path.dirname(os.path.dirname(args.allresdir))
    plot_grid(args.allresdir, args.graphmldir, args.outdir)
    plot_avenues_all(args.allresdir, args.graphmldir, args.outdir)


    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
