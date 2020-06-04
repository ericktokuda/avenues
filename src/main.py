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
from scipy.spatial import cKDTree

HOME = os.getenv('HOME')

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def parse_graphml(graphmlpath, pklpath, samplerad=-1):
    """Read graphml file to igraph object and dump it to @pklpath
    Receives the input path @graphmlpath and dump to @pklpath
    """
    info(inspect.stack()[0][3] + '()')

    if os.path.exists(pklpath):
        info('Loading existing file: {}'.format(pklpath))
        return pickle.load(open(pklpath, 'rb'))

    g = igraph.Graph.Read(graphmlpath)
    g.simplify(); g.to_undirected()
    g = sample_circle_from_graph(g, samplerad)
    pickle.dump(g, open(pklpath, 'wb'))
    return g

##########################################################
def sample_circle_from_graph(g, radius):
    """Short description
    Receives g, pklpath, overwrite=True and returns a ret
    """
    info(inspect.stack()[0][3] + '()')
    if radius < 0: return g
    coords = [(float(x), float(y)) for x, y in zip(g.vs['x'], g.vs['y'])]
    c0 = coords[np.random.randint(g.vcount())]
    ids = get_points_inside_region(coords, c0, radius)
    todel = np.ones(g.vcount(), bool)
    todel[ids] = False
    g.delete_vertices(np.where(todel == True)[0])
    return g

##########################################################
def get_points_inside_region(coords, c0, radius):
    """Get points from @df within circle of center @c0 and @radius
    """
    info(inspect.stack()[0][3] + '()')
    kdtree = cKDTree(coords)
    inds = kdtree.query_ball_point(c0, radius)
    return sorted(inds)

##########################################################
def analyze_random_increment_of_edges(g, nnewedges, outcsv):
    """Analyze random increment of n edges to @g for each value n in @nnewedges
    """
    info(inspect.stack()[0][3] + '()')
    acc = 0
    values = []

    for n in [0] + nnewedges:
        info('Adding {} edges'.format(n))
        nnew = n - acc
        for i in range(nnew):
            s, t = np.random.choice(g.vcount(), size=2, replace=False)
            g.add_edge(s, t)
        values.append([n, g.average_path_length()])
        acc += n

    df = pd.DataFrame(values, columns=['nnewedges', 'avgpathlen'])
    df.to_csv(outcsv, index=False)

##########################################################
def plot_map(g, outdir):
    """Do blah blah.
    Receives g, outdir and returns a ret
    """
    info(inspect.stack()[0][3] + '()')
    nrows = 1;  ncols = 1
    figscale = 8
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))
    lines = np.zeros((g.ecount(), 2, 2), dtype=float)

    for i, e in enumerate(g.es()):
        srcid = int(e.source)
        tgtid = int(e.target)

        lines[i, 0, 0] = g.vs[srcid]['x']
        lines[i, 0, 1] = g.vs[srcid]['y']
        lines[i, 1, 0] = g.vs[tgtid]['x']
        lines[i, 1, 1] = g.vs[tgtid]['y']
    lc = mc.LineCollection(lines, linewidths=0.5)
    axs[0, 0].add_collection(lc)
    axs[0, 0].autoscale()

    plt.savefig(pjoin(outdir, 'map.pdf'))

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    parser.add_argument('--samplerad', default=-1, type=float, help='Sample radius')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    np.random.seed(0)
    graphmlpath = 'data/map.graphml'
    pklpath = 'data/map.pkl'
    outcsv = pjoin(args.outdir, 'shortest.csv')
    nnewedges = [10]

    g = parse_graphml(graphmlpath, pklpath, samplerad=args.samplerad)
    info('nvertices: {}'.format(g.vcount()))
    info('nedges: {}'.format(g.ecount()))

    analyze_random_increment_of_edges(g, nnewedges, outcsv)
    
    plot_map(g, args.outdir)
    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

