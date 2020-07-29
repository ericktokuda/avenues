#!/usr/bin/env python3
"""Analysis of shortest paths in cities
We do not consider REAL distances, but difference in lat,lon
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
from enum import Enum
from itertools import combinations

HOME = os.getenv('HOME')

ORIGINAL = 0
BRIDGE = 1
BRIDGEACC = 2

#############################################################
def info(*args):
    pref = datetime.now().strftime('[%y%m%d %H:%M:%S]')
    print(pref, *args, file=sys.stdout)

##########################################################
def parse_graphml(graphmlpath, cachedir, undir=True, samplerad=-1):
    """Read graphml file to igraph object and dump it to @pklpath
    It gets the major component, and simplify it (neither multi nor self loops)
    Receives the input path @graphmlpath and dump to @pklpath.
    Assumes vertex attribs 'x' and 'y' are available
    """
    info(inspect.stack()[0][3] + '()')

    f = os.path.splitext(os.path.basename(graphmlpath))[0]
    if samplerad > -1: f += '_rad{}'.format(samplerad)
    pklpath = pjoin(cachedir, f + '.pkl')

    if os.path.exists(pklpath):
        info('Loading existing file: {}'.format(pklpath))
        return pickle.load(open(pklpath, 'rb'))

    g = igraph.Graph.Read(graphmlpath)
    g.simplify()

    if undir: g.to_undirected()

    g = sample_circle_from_graph(g, samplerad)

    g = g.components(mode='weak').giant()
    g['origvcount'] = g.vcount()
    info('g.is_connected():{}'.format(g.is_connected()))

    g = add_lengths(g)
    g.vs['type'] = ORIGINAL
    g.es['type'] = ORIGINAL
    g.es['bridgeid'] = -1
    pickle.dump(g, open(pklpath, 'wb'))
    return g

##########################################################
def sample_circle_from_graph(g, radius):
    """Sample a random region from the graph
    """
    info(inspect.stack()[0][3] + '()')
    
    if radius < 0: return g
    coords = [(x, y) for x, y in zip(g.vs['x'], g.vs['y'])]
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
def choose_bridge_endpoints(g):
    """Add @nnewedges to @g """
    # info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0]

    nvertices = len(orig)

    maxntries = 50
    for i in range(maxntries):
        s, t = np.random.randint(nvertices, size=2)
        if orig[t] not in g.neighbors(orig[s]):
            return np.array([[orig[s], orig[t]]])

    raise Exception('Too many tries on choosing a new edge\n' \
            'The graph is almost complete (v:{} e:{}) .'.format(g.vcount(), g.ecount()))

##########################################################
def calculate_edge_len(g, srcid, tgtid):
    """Calculate edge length based on 'x' and 'y' attributes"""
    src = np.array([float(g.vs[srcid]['x']), float(g.vs[srcid]['y'])])
    tgt = np.array([float(g.vs[tgtid]['x']), float(g.vs[tgtid]['y'])])
    return np.linalg.norm(tgt - src)

##########################################################
def add_wedge(g, srcid, tgtid, eid, bridgeid=-1):
    g.add_edge(srcid, tgtid, type=eid)
    eid = g.ecount() - 1
    g.es[eid]['length'] = calculate_edge_len(g, srcid, tgtid)
    g.es[eid]['bridgeid'] = bridgeid
    return g

##########################################################
def add_detour_route(g, edge, origtree, spacing, nnearest):
    """Add shortcut path etween @edge vertices"""
    # info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0]
    coords = origtree.data
    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    v = tgt - src
    vnorm = np.linalg.norm(v)
    versor = v / vnorm

    d = spacing

    lastpid = srcid
    vlast = srcid
    nnewedges = 0

    while d < vnorm:

        p = src + versor * d
        _, ids = origtree.query(p, 3) # in the worst case, the 2 first are the src and tgt
 
        for i, id in enumerate(ids):
            if orig[id] != srcid and orig[id] != tgtid:
                g = add_wedge(g, vlast, orig[id], BRIDGEACC)
                break

        vlast = id
        d += spacing

    return add_wedge(g, vlast, tgtid, BRIDGEACC)

##########################################################
def add_bridge(g, edge, origtree, spacing, nnearest):
    """Add @eid bridge and accesses in @g"""
    info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0]
    coords = origtree.data

    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    v = tgt - src
    vnorm = np.linalg.norm(v)
    versor = v / vnorm

    lastpid = srcid
    for d in np.arange(spacing, vnorm, spacing):
        p = src + versor * d
        params = {'type': BRIDGEACC, 'x': p[0], 'y': p[1]}
        g.add_vertex(p, **params) # new vertex in the bridge
        newvid = g.vcount() - 1
        g = add_wedge(g, lastpid, newvid, BRIDGE)
        g.vs[newvid]['x'] = p[0]
        g.vs[newvid]['y'] = p[1]
        _, ids = origtree.query(p, nnearest + 2)
 
        for i, id in enumerate(ids): # create accesses
            if i >= nnearest: break
            if orig[id] == srcid or orig[id] == tgtid: continue
            g = add_wedge(g, newvid, orig[id], BRIDGEACC)

        lastpid = newvid

    g = add_wedge(g, lastpid, tgtid, BRIDGE)
    return g

##########################################################
def partition_edges(g, endpoints, spacing, nnearest=1):
    """Partition bridges spaced by @spacing and each new vertex is connected to
    the nearest node
    """
    # info(inspect.stack()[0][3] + '()')

    vtypes = np.array(g.vs['type'])
    xs = np.array(g.vs['x'])
    ys = np.array(g.vs['y'])
    orig = list(np.where(vtypes != BRIDGEACC)[0])

    coords = [[float(xs[i]), float(ys[i])] for i in orig]

    origtree = cKDTree(coords)
    g = add_bridge(g, endpoints, origtree, spacing, nnearest)
    # add_detour_route(g, endpoints, origtree, spacing, nnearest)
        
    return g
       
##########################################################
def add_lengths(g):
    """Add lengths to the graph """
    info(inspect.stack()[0][3] + '()')
    for i, e in enumerate(g.es()):
        srcid, tgtid = e.source, e.target
        g.es[i]['length'] = calculate_edge_len(g, srcid, tgtid)
    return g

##########################################################
def calculate_avg_path_length(g, weighted=False, srctgttypes=None):
    """Calculate avg path length of @g.
    Calling all at once, without the loop on the vertices, it crashes
    for large graphs """
    # info(inspect.stack()[0][3] + '()')

    if g.is_directed():
        raise Exception('This method considers an undirected graph')

    weights = g.es['length'] if weighted else np.array([1] * g.ecount())

    dsum = 0; d2sum = 0
    if srctgttypes == None:
        vids = list(range(g.vcount()))
    else: # consider src and tget of particular types
        vids = np.nonzero(np.isin(np.array(g.vs['type']), srctgttypes))[0]
    
    for i, srcid in enumerate(range(len(vids))): #Assuming undirected graph
        aux =  np.array(g.shortest_paths(source=vids[i], target=vids[i+1:],
            weights=g.es['length'], mode='ALL'))

        dsum += np.sum(aux)
        d2sum += np.sum(np.square(aux))

    ndists = int((g.vcount() * (g.vcount()-1)) / 2) # diagonal
    distmean = dsum / ndists
    diststd = np.sqrt(( d2sum - (dsum**2)/ndists ) / ndists)
    return distmean, diststd

##########################################################
def extract_features(g, nbridges):
    """Extract features from graph @g """
    info(inspect.stack()[0][3] + '()')
    etypes = np.array(g.es['type'])
    meanw, stdw = calculate_avg_path_length(g, weighted=True,
            srctgttypes=[ORIGINAL, BRIDGE],)

    bv = g.betweenness()
    betwvmean, betwvstd = np.mean(bv), np.std(bv)

    return [g.vcount(), g.ecount(),
        nbridges, len(np.where(etypes == BRIDGEACC)[0]),
        meanw, stdw, betwvmean, betwvstd,
        ]
##########################################################
def analyze_increment_of_random_edges(g, bridges, spacing, outcsv):
    """Analyze increment of @bridges to @g"""
    info(inspect.stack()[0][3] + '()')

    g.es['type'] = ORIGINAL
    prev = 0

    data = []
    data.append(extract_features(g, 0))
    nbridges = len(bridges)

    for i in range(1, nbridges + 1):
        info('bridgeid:{}'.format(i))
        es = bridges[i-1]
        g.vs[es[0]]['type'] = BRIDGE
        g.vs[es[1]]['type'] = BRIDGE
        g = partition_edges(g, es, spacing, nnearest=1)
        data.append(extract_features(g, i))

    cols = 'nvertices,nedges,nbridges,naccess,' \
            'avgpathlen,stdpathlen,betwvmean,betwvstd'.\
            split(',')
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(outcsv, index=False)
    info('df:{}'.format(df))
    return g

##########################################################
def hex2rgb(hexcolours, normalized=False, alpha=None):
    rgbcolours = np.zeros((len(hexcolours), 3), dtype=int)
    for i, h in enumerate(hexcolours):
        rgbcolours[i, :] = np.array([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])

    if alpha != None:
        aux = np.zeros((len(hexcolours), 4), dtype=float)
        aux[:, :3] = rgbcolours / 255
        aux[:, -1] = alpha # alpha
        rgbcolours = aux
    elif normalized:
        rgbcolours = rgbcolours.astype(float) / 255

    return rgbcolours

##########################################################
def plot_map(g, outdir, vertices=False):
    """Plot map g, according to 'type' attrib both in vertices and in edges
    """
    
    info(inspect.stack()[0][3] + '()')
    nrows = 1;  ncols = 1
    figscale = 5
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))
    lines = np.zeros((g.ecount(), 2, 2), dtype=float)
    
    ne = g.ecount()
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    palettergb = hex2rgb(palettehex, normalized=True, alpha=0.7)

    ecolours = [ palettergb[i, :] for i in g.es['type']]

    for i, e in enumerate(g.es()):
        srcid = int(e.source)
        tgtid = int(e.target)

        lines[i, 0, :] = [g.vs[srcid]['x'], g.vs[srcid]['y']]
        lines[i, 1, :] = [g.vs[tgtid]['x'], g.vs[tgtid]['y']]

    lc = mc.LineCollection(lines, colors=ecolours, linewidths=figscale*.1)
    axs[0, 0].add_collection(lc)
    axs[0, 0].autoscale()

    coords = np.array([[float(x), float(y)] for x, y in zip(g.vs['x'], g.vs['y'])])

    if vertices:
        vids = np.where(np.array(g.vs['type']) == ORIGINAL)[0]
        axs[0, 0].scatter(coords[vids, 0], coords[vids, 1])

    vids = np.where(np.array(g.vs['type']) == BRIDGE)[0]
    axs[0, 0].scatter(coords[vids, 0], coords[vids, 1], s=figscale*.1, c='k')

    plt.savefig(pjoin(outdir, 'map.pdf'))

##########################################################
def choose_new_edges(g, nnewedges):
    """Randomly choose new edges. Multiple edges are not allowed"""
    info(inspect.stack()[0][3] + '()')

    all = set(list(combinations(np.arange(g.vcount()), 2)))
    es = [[e.source, e.target] for e in list(g.es)]

    if not g.is_directed():
        y0 = np.min(es, axis=1).reshape(-1, 1)
        y1 = np.max(es, axis=1).reshape(-1, 1)
        es = np.concatenate([y0, y1], axis=1)
    es = set([(e[0], e[1]) for e in es]) # For faster checks

    available = np.array(list(all.difference(es)))
    inds = np.random.randint(len(available), size=nnewedges) # random

    return available[inds]

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphml', required=True,
            help='Path to the map in graphml')
    parser.add_argument('--samplerad', default=-1, type=float,
            help='Sample radius')
    parser.add_argument('--nedges', default=1, type=int,
            help='Sample radius')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    np.random.seed(0)
    outcsv = pjoin(args.outdir, 'results.csv')
    outpklpath = pjoin(args.outdir, 'finalgraph.pkl')
    nnewedges = args.nedges
    maxnedges = np.max(nnewedges)

    g = parse_graphml(args.graphml, args.outdir, undir=True,
            samplerad=args.samplerad)
    spacing = np.mean(g.es['length']) * 10

    info('nvertices: {}'.format(g.vcount()))
    info('nedges: {}'.format(g.ecount()))

    es = choose_new_edges(g, nnewedges)
    g = analyze_increment_of_random_edges(g, es, spacing, outcsv)

    pickle.dump(g, open(outpklpath, 'wb'))

    plot_map(g, args.outdir)
    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

