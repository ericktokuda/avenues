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

    # graphsdir = '/home/frodo/results/graffiti/20200202-types/'
    # graphmlpath = '/home/frodo/results/graffiti/20200202-types/20200221-citysp.graphml'
    # g = igraph.Graph.Read(graphmlpath)
    # g.simplify()
    # g.to_undirected()
    # coords = [(float(x), -float(y)) for x, y in zip(g.vs['x'], g.vs['y'])]
    # pklpath = '/tmp/citysp.pkl'
    # pickle.dump(g, open(pklpath, 'wb'))

    pklpath = pjoin(HOME, 'temp/citysp.pkl')
    g = pickle.load(open(pklpath, 'rb'))
    info('g.vcount():{}'.format(g.vcount()))
    g.delete_vertices(np.arange(100000))

    # breakpoint()
    
    norig = g.ecount()
    acc = 0
    values = []
    for m in [10, 100, 1000]:
    # for x in [10, 100, 1000, 10000]:
        info('m:{}'.format(m))
        nnew = m - acc
        for i in range(nnew):
            s, t = np.random.choice(g.vcount(), size=2, replace=False)
            g.add_edge(s, t)
        values.append([m, g.average_path_length()])
        acc += m

    df = pd.DataFrame(values, columns=['nnewedges', 'avgpathlen'])
    df.to_csv(pjoin(args.outdir, 'lengths.csv'))
    
    # plot_map(g, args.outdir)
    info('Elapsed time:{}'.format(time.time()-t0))

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
if __name__ == "__main__":
    main()

