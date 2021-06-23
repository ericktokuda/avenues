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

    city = os.path.splitext(f)[0]

    return [degrmean, degrstd, fr[3], fr[4], fr[5]]

##########################################################
def extract_angle_entropy(g, bins):
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

    distr, _ = np.histogram(angles, bins=bins, density=True)
    return -(distr*np.log(np.abs(distr))).sum()

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphsdir', required=True, help='Path to the dir with graphs inside')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)
    csvpath = pjoin(args.outdir, 'stats.csv')

    files = sorted(os.listdir(args.graphsdir))
    data = []

    # Angle feature params
    bins = np.arange(-np.pi/2, +np.pi/2, np.pi/6)

    # Node entropy params
    sigma = 1000        # Std (in meters) used for the gaussian kernel
    max_ratio = 0.2     # Thresh for high dens. regions (rel. to maximum)
    min_width = 100     # Min. width (in meters) allowed for the urban area

    # Lacunarity params
    px_per_km = 100 # To obtain per meter, divide by 1000
    max_radius = 100
    delta_radius = 5

    for f in files:
        graphml = pjoin(args.graphsdir, f)
        if not graphml.endswith('.graphml'): continue
        info('f:{}'.format(f))
        g = graph.simplify_graphml(graphml)
        # degrfeats = extract_degree_feats(g)
        entropyan = extract_angle_entropy(g, bins)
        entropyvx = calculate_rastered_graph_entropy(graphml, sigma, max_ratio,
                                                    min_width)
        radii, lacun = calculate_lacunarity(graphml, max_radius, delta_radius,
                                            px_per_km)
        breakpoint()
        # row = angfeats + degrfeats

    # data.append([city, degrmean, degrstd, fr[3], fr[4], fr[5]])
    df = pd.DataFrame(data, columns=['city', 'degrmean', 'degrstd',
                                     'frac3', 'frac4', 'frac5'])
    df.to_csv(pjoin(args.outdir, 'feats.csv'), index=False)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
