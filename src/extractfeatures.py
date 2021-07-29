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
def extract_degree_feats(g):
    """Calculate features related to the degree distribution. Extract the
    mean, std, fraction of vertices with degrees 3, 4 and 5."""
    info(inspect.stack()[0][3] + '()')
    degrees = np.array(g.degree())
    n = len(degrees)
    degrmean = np.mean(degrees)
    degrstd = np.std(degrees)
    uvals, ucounts = np.unique(degrees, return_counts=True)

    fr = {}
    for k in [3, 4, 5]:
        ids = np.where(uvals == k)[0]
        if len(ids) == 0:
            fr[k] = 0
        else:
            idx = ids[0]
            fr[k] = ucounts[idx] / n

    counts, _ = np.histogram(degrees)
    distr = counts / np.sum(counts)
    positive = distr[distr > 0]
    degrentr = -(positive*np.log(np.abs(positive))).sum()
    return [degrmean, degrstd, degrentr, fr[3], fr[4], fr[5]]

##########################################################
def extract_clucoeff_feats(g):
    """Calculate clustering coefficient entropy """
    info(inspect.stack()[0][3] + '()')
    clucoeffs = np.array(g.as_undirected().transitivity_local_undirected())
    valid = np.argwhere(~np.isnan(clucoeffs)).flatten()
    counts, _ = np.histogram(valid)
    distr = counts / np.sum(counts)
    return -(distr*np.log(np.abs(distr))).sum()

##########################################################
def calculate_angle_entropy(g, bins):
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

    counts, _ = np.histogram(angles, bins=bins)
    distr = counts / np.sum(counts)
    return -(distr*np.log(np.abs(distr))).sum()

##########################################################
def calculate_accessib_feats(accpath):
    """Calculate accessibility features"""
    info(inspect.stack()[0][3] + '()')
    accs = np.loadtxt(accpath)
    counts, _ = np.histogram(accs)
    distr = counts / np.sum(counts)
    positive = distr[distr > 0]
    accentr = -(positive*np.log(np.abs(positive))).sum()
    return [np.mean(accs), np.std(accs), accentr]

##########################################################
def main(graphsdir, accessibdir, outdir):
    """Main function"""
    info(inspect.stack()[0][3] + '()')
    outpath = pjoin(outdir, 'feats.csv')

    files = sorted(os.listdir(graphsdir))
    data = []

    # Angle feature params
    bins = np.arange(-np.pi/2, +np.pi/2, np.pi/6)

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

    data = []
    for f in files:
        graphml = pjoin(graphsdir, f)
        if not graphml.endswith('.graphml'): continue
        city = f.replace('.graphml', '')
        info('{}'.format(f))
        accpath = pjoin(accessibdir, city + accsuf)
        if not os.path.exists(accpath):
            info('Accessib file {} does not exist. Skipping'.format(accpath))
            continue
        g = graph.simplify_graphml(graphml)
        degrfeats = extract_degree_feats(g)
        clustfeats = extract_clucoeff_feats(g)
        entropyan = calculate_angle_entropy(g, bins)
        entropyvx = calculate_rastered_graph_entropy(graphml, sigma, max_ratio,
                                                    min_width)
        radii, lacun = calculate_lacunarity(graphml, max_radius, delta_radius,
                                            px_per_km)
        accfeats = calculate_accessib_feats(accpath)
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
