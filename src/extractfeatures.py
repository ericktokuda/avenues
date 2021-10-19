#!/usr/bin/env python3
"""Extract topological features"""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from myutils import info, create_readme, append_to_file, graph
from ventropy import calculate_rastered_graph_entropy
from lacuncaller import calculate_lacunarity

##########################################################
def extract_degree_feats(g, binsz):
    """Calculate features related to the degree distribution. Extract the
    mean, std, fraction of vertices with degrees 3, 4 and 5."""
    info(inspect.stack()[0][3] + '()')
    degrees = np.array(g.degree())
    n = len(degrees)
    degrmean = np.mean(degrees)
    degrstd = np.std(degrees)
    uvals, ucounts = np.unique(degrees, return_counts=True)

    nbins = int((np.max(degrees) - np.min(degrees)) / binsz) + 1
    fr = {}
    for k in [3, 4, 5]:
        ids = np.where(uvals == k)[0]
        if len(ids) == 0:
            fr[k] = 0
        else:
            idx = ids[0]
            fr[k] = ucounts[idx] / n

    counts, _ = np.histogram(degrees, bins=nbins)
    distr = counts / np.sum(counts)
    positive = distr[distr > 0]
    degrentr = -(positive*np.log(np.abs(positive))).sum()
    return [degrmean, degrstd, degrentr, fr[3], fr[4], fr[5]]

##########################################################
def extract_clucoeff_feats(g, binsz):
    """Calculate clustering coefficient entropy """
    info(inspect.stack()[0][3] + '()')
    clucoeffs = np.array(g.as_undirected().transitivity_local_undirected())
    valid = np.argwhere(~np.isnan(clucoeffs)).flatten()
    nbins = int((np.max(valid) - np.min(valid)) / binsz) + 1
    counts, _ = np.histogram(valid, bins=nbins)
    distr = counts / np.sum(counts)
    return -(distr*np.log(np.abs(distr))).sum()

##########################################################
def calculate_angle_entropy(g, binsz):
    """Extract orientation from each edge. We adopt the angle 0 from a given point as the horizontal vector pointing to the right. Angles increase increase anti-clockwise.
    """
    coords = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
    angles = - np.ones(g.ecount(), dtype=float)
    for i, e in enumerate(g.es()):
        s, t = e.source, e.target
        sx, sy = coords[s, :]
        tx, ty = coords[t, :]
        if tx - sx == 0: angles[i] = 0
        else: angles[i] = np.arctan((ty - sy) / (tx - sx))

    nbins = int((np.max(angles) - np.min(angles)) / binsz) + 1
    counts, _ = np.histogram(angles, bins=nbins)
    distr = counts / np.sum(counts)
    return -(distr*np.log(np.abs(distr))).sum()

##########################################################
def calculate_accessib_feats(accpath, binsz):
    """Calculate accessibility features"""
    info(inspect.stack()[0][3] + '()')
    accs = np.loadtxt(accpath)
    nbins = int((np.max(accs) - np.min(accs)) / binsz) + 1
    counts, _ = np.histogram(accs)
    distr = counts / np.sum(counts)
    positive = distr[distr > 0]
    accentr = -(positive*np.log(np.abs(positive))).sum()
    return [np.mean(accs), np.std(accs), accentr]

##########################################################
def compare_min_max(values, minmax):
    """It expects an array of values and a minmax. Returns the new min and max"""
    min_= np.min(values) if np.min(values) < minmax[0] else minmax[0]
    max_ = np.max(values) if np.max(values) > minmax[1] else minmax[1]
    return min_, max_

##########################################################
def get_ranges(gfiles, graphsdir, accessibdir, accsuf):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    keys = ['degree', 'trans', 'angle', 'posx', 'posy', 'accessib']
    ranges = { k: [9999999, -9999999] for k in keys}
    ranges['angle'] = [-np.pi/2, +np.pi/2]

    for f in gfiles:
        graphml = pjoin(graphsdir, f)
        city = f.replace('.graphml', '')
        info('{}'.format(f))
        accpath = pjoin(accessibdir, city + accsuf)
        g = graph.simplify_graphml(graphml)

        degrees = g.degree()
        ranges['degree'] = compare_min_max(degrees, ranges['degree'])

        trans = np.array(g.as_undirected().transitivity_local_undirected())
        trans = np.argwhere(~np.isnan(trans)).flatten()
        ranges['trans'] = compare_min_max(trans, ranges['trans'])

        import osmnx as ox
        from ventropy import get_pos_array
        graphox = ox.load_graphml(graphml)
        graphox = ox.project_graph(graphox)
        pos = get_pos_array(graphox)
        posx, posy = pos.T
        ranges['posx'] = compare_min_max(posx, ranges['posx'])
        ranges['posy'] = compare_min_max(posy, ranges['posy'])

        accs = np.loadtxt(accpath)
        ranges['accessib'] = compare_min_max(accs, ranges['accessib'])

    return ranges

##########################################################
def get_binszs(ranges, angledelta):
    """Get binszs according to the ranges"""
    binszs = {}
    binszs['degree'] = 1
    binszs['angle'] = angledelta
    binszs['trans'] = (ranges['trans'][1] - ranges['trans'][0]) / 2
    binszs['accessib'] = (ranges['accessib'][1] - ranges['accessib'][0]) / 2
    x =  ranges['posx'][1] - ranges['posx'][0] / 2
    y =  ranges['posy'][1] - ranges['posy'][0] / 2
    binszs['pos'] = min(x, y)

    return binszs
##########################################################
def main(graphsdir, accessibdir, outdir):
    """Main function"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'feats.csv')

    files = sorted(os.listdir(graphsdir))
    data = []

    # Angle feature params
    angledelta = np.pi / 6
    bins = np.arange(-np.pi/2, +np.pi/2, angledelta)

    # Node entropy params
    sigma = 1000        # Std (in meters) used for the gaussian kernel
    max_ratio = 0.2     # Thresh for high dens. regions (rel. to maximum)
    min_width = 100     # Min. width (in meters) allowed for the urban area

    # Lacunarity params
    px_per_km = 100 # To obtain per meter, divide by 1000
    max_radius = 52
    delta_radius = 10

    accstep = 5
    accsuf = '_undirected_acc{:02d}.txt'.format(accstep)

    # ranges = {'}
    gfiles = []
    for f in files:
        if not f.endswith('.graphml'): continue
        city = f.replace('.graphml', '')
        accpath = pjoin(accessibdir, city + accsuf)
        if not os.path.exists(accpath):
            info('Accessib file {} does not exist. Skipping'.format(accpath))
            continue
        gfiles.append(f)

    ranges = get_ranges(gfiles, graphsdir, accessibdir, accsuf)
    binszs = get_binszs(ranges, angledelta)

    data = []
    for f in gfiles:
        graphml = pjoin(graphsdir, f)
        city = f.replace('.graphml', '')
        info('{}'.format(f))
        accpath = pjoin(accessibdir, city + accsuf)
        g = graph.simplify_graphml(graphml)
        degrfeats = extract_degree_feats(g, binszs['degree'])
        clustfeats = extract_clucoeff_feats(g, binszs['trans'])
        entropyan = calculate_angle_entropy(g, binszs['angle'])
        entropyvx = calculate_rastered_graph_entropy(graphml, sigma, max_ratio,
                                                    binszs['pos'], min_width)
        radii, lacun = calculate_lacunarity(graphml, max_radius, delta_radius,
                                            px_per_km)
        accfeats = calculate_accessib_feats(accpath, binszs['accessib'])
        row = [city] + degrfeats + [clustfeats] + \
            [entropyan, entropyvx] + list(lacun) + accfeats
        data.append(row)

    lacuncols = ['lacun{}'.format(r) for r in radii]
    cols = ['city', 'degrmean', 'degrstd', 'degrentr', 'degr3', 'degr4', 'degr5',
            'clucoeff', 'entropyang', 'entropyvx'] + lacuncols + \
        ['accmean', 'accstd', 'accentr']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(outpath, index=False)

##########################################################
if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphsdir', required=True, help='Path to the dir with graphs inside')
    parser.add_argument('--accessibdir', required=True, help='Path to the dir with accessibilities inside')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.graphsdir, args.accessibdir, args.outdir)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
