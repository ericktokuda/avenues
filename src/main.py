#!/usr/bin/env python3
"""Analysis of shortest paths in cities as we include shortcut connections.
We expect a network in graphml format with x,y attributes representing lon, lat
and we incrementally add bridges (or detour routes) and calculate the
shortest path lengths."""

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import igraph
import pickle
import pandas as pd
from scipy.spatial import cKDTree
from itertools import combinations
from myutils import info, graph, plot, geo
from sklearn.neighbors import BallTree

ORIGINAL = 0
BRIDGE = 1
BRIDGEACC = 2

##########################################################
def parse_graphml(graphmlpath, undir=True, samplerad=-1):
    """Read graphml file to igraph object and dump it to @pklpath
    It gets the major component, and simplify it (neither multi nor self loops)
    Receives the input path @graphmlpath and dump to @pklpath.
    Assumes vertex attribs 'x' and 'y' are available """
    info(inspect.stack()[0][3] + '()')

    g = graph.simplify_graphml(graphmlpath, directed=False)
    g['vcountorig'] = g.vcount() #TODO: move this to myutils
    g['ecountorig'] = g.ecount()
    g = sample_circle_from_graph(g, samplerad)
    g = g.components(mode='weak').giant()
    g['vcountsampled'] = g.vcount()
    g['ecountsampled'] = g.ecount()

    g.vs['type'] = ORIGINAL
    g.es['type'] = ORIGINAL
    g.es['bridgeid'] = -1

    origid2sampleid = {}
    for i, origid in enumerate(g.vs['origid']):
        origid2sampleid[origid] = i

    g['origid2sampleid'] = origid2sampleid
    g['coords'] = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
    
    return g

##########################################################
def induced_by(gorig, vs):
    g = gorig.copy()
    todel = np.ones(g.vcount(), bool)
    todel[vs] = False
    g.delete_vertices(np.where(todel == True)[0])
    return g

##########################################################
def sample_circle_from_graph(g, radius):
    """Sample a random region from the graph """
    info(inspect.stack()[0][3] + '()')
    
    if radius < 0: return g
    coords = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
    c0 = coords[np.random.randint(g.vcount())]
    ids = get_points_inside_region(coords, c0, radius)
    return induced_by(g, ids)

##########################################################
def get_points_inside_region(coords, c0, radius):
    """Get points from @df within circle of center @c0 and @radius"""
    info(inspect.stack()[0][3] + '()')

    bt = BallTree(np.deg2rad(coords), metric='haversine')
    center = np.deg2rad(np.array([c0]))
    inds, dists = bt.query_radius(center, radius / geo.R,
            return_distance=True, sort_results=True)
    return inds[0]

##########################################################
def calculate_edge_len(g, srcid, tgtid):
    """Calculate edge length based on 'x' and 'y' attributes"""
    src = np.array([float(g.vs[srcid]['x']), float(g.vs[srcid]['y'])])
    tgt = np.array([float(g.vs[tgtid]['x']), float(g.vs[tgtid]['y'])])
    return np.linalg.norm(tgt - src)

##########################################################
def add_wedge(g, srcid, tgtid, etype, bridgeid=-1):
    g.add_edge(srcid, tgtid, type=etype)
    eid = g.ecount() - 1
    lon1, lat1 = float(g.vs[srcid]['x']), float(g.vs[srcid]['y'])
    lon2, lat2 = float(g.vs[tgtid]['x']), float(g.vs[tgtid]['y'])
    g.es[eid]['length'] = geo.haversine(lon1, lat1, lon2, lat2)
    g.es[eid]['bridgeid'] = bridgeid
    return g

##########################################################
def add_detour_route(g, edge, origtree, spacing, nnearest):
    """Add shortcut path between @edge vertices"""
    info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0]
    coords = origtree.data
    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    vnorm = geo.haversine(src[0], src[1], tgt[0], tgt[1])
    versor = v / vnorm

    d = spacing

    lastpid = srcid
    vlast = srcid
    nnewedges = 0

    while d < vnorm:
        p = src + versor * d # in the worst case,
        _, ids = origtree.query(p, 3) # the 2 first are the src and tgt
 
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
    vnorm = geo.haversine(src[0], src[1], tgt[0], tgt[1])
    versor = v / vnorm

    lastpid = srcid
    
    for d in np.arange(spacing, vnorm, spacing):
        p = src + versor * d
        params = {'type': BRIDGEACC, 'x': p[0], 'y': p[1]}
        g.add_vertex(p, **params) # new vertex in the bridge
        newvid = g.vcount() - 1
        nadded = len(np.where(np.array(g.vs['type']) == BRIDGEACC)[0])
        g.vs[newvid]['origid'] = g['vcountsampled'] + nadded
        
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
    the nearest node """
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
def calculate_avg_path_length(g, weighted=False):
    """Calculate avg path length of @g.
    Calling all at once, without the loop on the vertices, it crashes
    for large graphs """
    # info(inspect.stack()[0][3] + '()')

    pathlens = {}

    if g.is_directed():
        raise Exception('This method considers an *undirected* graph')
    elif g.vcount() < 2: return 0, 0

    weights = g.es['length'] if weighted else np.array([1] * g.ecount())

    for srcid in range(g['vcountsampled']): #Assuming an undirected graph
        tgts = list(range(srcid + 1, g['vcountsampled']))
        spaths =  g.shortest_paths(source=srcid, target=tgts,
            weights=g.es['length'], mode='ALL')

        
        for tgtid, l in zip(tgts, spaths[0]): pathlens[(srcid, tgtid)] = l

    return pathlens

##########################################################
def calculate_local_avgpathlen(gorig, balls, dists, srctgttypes):
    """Average path length in a small ball with radius given by @scale"""
    # info(inspect.stack()[0][3] + '()')

    # Find the vertices with the desired types
    induced = induced_by(gorig, vs)
    # coords = np.array([[x, y] for x, y in zip(g.vs['x'], g.vs['y'])])
    # coords = get_coords(g)

    meanw = np.zeros(len(refvids), dtype=float)
    # stdw = np.zeros(g.vcount(), dtype=float)

    for i, vid in enumerate(refvids):
        # bt = BallTree(np.deg2rad(coords), metric='haversine')
        c0 = np.array([g['coords'][vid, :]])
        inds = tree.query_radius(np.deg2rad(c0), scale / geo.R)

        if len(inds) == 0: continue

        inds = sorted(inds[0])

        m = len(inds)
        d = np.zeros(int(m * (m-1) / 2))
        k = 0

        for j in range(len(inds) - 1):
            sorig = g.vs[inds[j]]['origid']
            for t in inds[j+1:]:
                torig = g.vs[t]['origid']
                if torig == None: breakpoint()
                
                d[k] = dists[(sorig, torig)]
                k += 1
        
        meanw[i] = np.mean(d)

        # induced = induced_by(g, inds[0])
        # induced = induced.components(mode='weak').giant()
        # _, meanw[i], _ = calculate_avg_path_length(induced, weighted=True,
                # srctgttypes=[ORIGINAL, BRIDGE], cached=dists)

    return meanw

def calculate_local_avgpathlen3(g, ballids, pathlens):
    from itertools import combinations
    combs = list(combinations(ballids, 2))
    pathlens = [ pathlens[comb] for comb in combs ]
    return np.mean(pathlens), np.std(pathlens)
    

##########################################################
def extract_features_global(g, pathlens):
    """Extract graph global measurements"""
    info(inspect.stack()[0][3] + '()')

    bv = g.betweenness()
    betwvmean, betwvstd = np.mean(bv), np.std(bv)

    pathlensv = np.array(list(pathlens.values()))
    
    return np.mean(pathlensv), np.std(pathlensv), betwvmean, betwvstd

##########################################################
def extract_features_local(g, ballids, pathlens):
    """Extract local features from ballids"""

    pathsmean, pathsstd = calculate_local_avgpathlen3(g, ballids, pathlens)
    # other features

    return pathsmean

##########################################################
def extract_features(g, balls, nbridges):
    """Extract features from graph @g """
    # info(inspect.stack()[0][3] + '()')
    etypes = np.array(g.es['type'])

    pathlens = calculate_avg_path_length(g, weighted=True)

    pathsmean, pathsstd, betwvmean, betwvstd = \
            extract_features_global(g, pathlens)
    
    for ball in balls:
        lpathsmean, lpathsstd = extract_features_local(g, ball, pathlens)

    # return [g.vcount(), g.ecount(),
        # nbridges, len(np.where(etypes == BRIDGEACC)[0]),
        # meanw, stdw, betwvmean, betwvstd, meanlw, stdlw]

    # ret = [g.vcount(), g.ecount(),
        # nbridges, len(np.where(etypes == BRIDGEACC)[0]),
        # meanw, stdw, betwvmean, betwvstd]
    # return ret + meanlws.tolist()

##########################################################
def analyze_increment_of_bridges(g, bridges, spacing, balls, outcsv):
    """Analyze increment of @bridges to @g. We add entrances/exit spaced
    by @spacing and output to @outcsv."""
    info(inspect.stack()[0][3] + '()')

    g.es['type'] = ORIGINAL
    prev = 0

    features = [ extract_features(g, balls, 0) ]

    nbridges = len(bridges)

    for i in range(1, nbridges + 1):
        info('bridgeid:{}'.format(i))
        es = bridges[i-1]
        g.vs[es[0]]['type'] = BRIDGE
        g.vs[es[1]]['type'] = BRIDGE
        g = partition_edges(g, es, spacing, nnearest=1)

        newfeatures = extract_features(g, refvids, tree, scale, i)
        features.append(newfeatures)

    # cols = 'nvertices,nedges,nbridges,naccess,' \
            # 'avgpathlen,stdpathlen,betwvmean,betwvstd,' \
            # 'lavgpathlen,lstdpathlen'.\
            # split(',')
    
    refstr = ','.join([ 'ref{:02d}'.format(j) for j in range(nref)])
    cols = 'nvertices,nedges,nbridges,naccess,' \
            'avgpathlen,stdpathlen,betwvmean,betwvstd,' + refstr
    cols = cols.split(',')
            # 'lavgpathlen0,lavgpathlen1'.\
            # split(',')
    df = pd.DataFrame(features, columns=cols)
    df.to_csv(outcsv, index=False)
    info('df:{}'.format(df))
    return g

##########################################################
def plot_map(g, outdir, vertices=False):
    """Plot map g, according to 'type' attrib both in vertices and in edges """
    
    info(inspect.stack()[0][3] + '()')
    nrows = 1;  ncols = 1
    figscale = 5
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                figsize=(ncols*figscale, nrows*figscale))
    lines = np.zeros((g.ecount(), 2, 2), dtype=float)
    
    ne = g.ecount()
    palettehex = plt.rcParams['axes.prop_cycle'].by_key()['color']
    palettergb = plot.hex2rgb(palettehex, normalize=True, alpha=0.7)

    ecolours = [ palettergb[i, :] for i in g.es['type']]

    for i, e in enumerate(g.es()):
        srcid = int(e.source)
        tgtid = int(e.target)

        lines[i, 0, :] = [g.vs[srcid]['x'], g.vs[srcid]['y']]
        lines[i, 1, :] = [g.vs[tgtid]['x'], g.vs[tgtid]['y']]

    lc = mc.LineCollection(lines, colors=ecolours, linewidths=figscale*.1)
    axs[0, 0].add_collection(lc)
    axs[0, 0].autoscale()

    # coords = np.array([[float(x), float(y)] for x, y in zip(g.vs['x'], g.vs['y'])])
    # coords = get_coords(g)


    if vertices:
        vids = np.where(np.array(g.vs['type']) == ORIGINAL)[0]
        axs[0, 0].scatter(g['coords'][vids, 0], g['coords'][vids, 1])

    vids = np.where(np.array(g.vs['type']) == BRIDGE)[0]
    axs[0, 0].scatter(g['coords'][vids, 0], g['coords'][vids, 1], s=figscale*.1, c='k')
    # axs[0, 0].set_xticks([])
    # axs[0, 0].set_yticks([])

    plt.savefig(pjoin(outdir, 'map.pdf'))

##########################################################
def choose_bridges_random(g, nnewedges, available):
    inds = np.random.randint(len(available), size=nnewedges) # random
    return available[inds]

##########################################################
def choose_bridges_random_minlen(g, nnewedges, available, minlen):
    coords = g['coords']
    inds = np.arange(len(available))
    np.random.shuffle(inds)

    bridges = []
    for ind in inds:
        srcid, tgtid = available[ind]
        l = geo.haversine(coords[srcid][0], coords[srcid][1],
                coords[tgtid][0], coords[tgtid][1])
        
        if l > minlen: bridges.append(available[ind])
        if len(bridges) == nnewedges: break

    return bridges

##########################################################
def choose_bridges_acc(g, nnewedges, available):
    accpath = '/home/frodo/results/bridges/20200729-accessibs/sp_undirected_acc05.txt'
    acc = [float(x) for x in open(accpath).read().strip().split('\n')]
    g.vs['acc'] = acc
    aux = np.argsort(g.vs['acc']) # accessibility
    quant = int(g.vcount() * .25)
    if quant < 2*nnewedges: quant *= 2
    srcs = aux[:quant]
    tgts = aux[-quant:]

    bridges = []
    for i in range(quant):
        if len(bridges) == nnewedges: return bridges
        if [srcs[i], tgts[i]] not in available: continue
        bridges.append([srcs[i], tgts[i]])
    return bridges

##########################################################
def choose_new_bridges(g, nnewedges, minlen=-1):
    """Choose new edges. Multiple edges are not allowed.
    We compute the set difference between all possible edges
    and the existing ones."""
    info(inspect.stack()[0][3] + '()')

    # All possible edges
    all = set(list(combinations(np.arange(g.vcount()), 2)))
    es = [[e.source, e.target] for e in list(g.es)]

    if not g.is_directed():
        y0 = np.min(es, axis=1).reshape(-1, 1)
        y1 = np.max(es, axis=1).reshape(-1, 1)
        es = np.concatenate([y0, y1], axis=1)
    es = set([(e[0], e[1]) for e in es]) # For faster checks

    available = np.array(list(all.difference(es))) # all - existing

    # return choose_bridges_acc(g, available)
    # return choose_bridges_random(g, nnewedges, available)
    return choose_bridges_random_minlen(g, nnewedges,
            available, minlen)

##########################################################
def sampleid2origid(g, sampleids):
    return g.vs[list(sampleids)]['origid']

##########################################################
def origid2sampleid(g, origids):
    return [ g.vs['origid2sampleid'][list(origid)] for origid in origids ]

##########################################################
# def get_neighbourhoods(centerids, coords, r):
def get_neighbourhoods(g, centerids, r):
    """Get neighbour ids within radius r for each c0 in c0s.
    It includes self."""
    info(inspect.stack()[0][3] + '()')
    coords = g['coords']
    tree = BallTree(np.deg2rad(coords), metric='haversine')

    neighbours = []
    for id in centerids: #TODO: remove loop
        center = np.deg2rad(np.array([coords[id, :]]))
        inds = tree.query_radius(center, r / geo.R)
        neighbours.append(sorted(inds[0]))

    return neighbours

##########################################################
def get_neighbourhoods_origids(g, centerids, r):
    """Get original id of the neighbours within radius r for each center.
    It includes self."""
    neids = get_neighbourhoods(centerids, g['coords'], r)
    return [ np.array(g.vs['origid'])[ids] for ids in neids ]

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphml', required=True,
            help='Path to the map in graphml')
    parser.add_argument('--samplerad', default=-1, type=float,
            help='Sample radius')
    parser.add_argument('--nbridges', default=1, type=int,
            help='Number of shortcut connections')
    parser.add_argument('--outdir', default='/tmp/out/',
            help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    np.random.seed(0)
    outcsv = pjoin(args.outdir, 'results.csv')
    outpklpath = pjoin(args.outdir, 'finalgraph.pkl')
    maxnedges = np.max(args.nbridges)

    minlen = 1
    spacing = .2
    scale = .5

    g = parse_graphml(args.graphml, undir=True, samplerad=args.samplerad)
    
    info('sampled nvertices: {}'.format(g.vcount()))
    info('sampled nedges: {}'.format(g.ecount()))

    es = choose_new_bridges(g, args.nbridges, minlen)

    nref = 30
    centerids = np.random.permutation(g.vcount())[:nref]
    balls = get_neighbourhoods(g, centerids, scale)
    # balls = get_neighbourhoods_origids(g, centerids, scale)

    g = analyze_increment_of_bridges(g, es, spacing, balls, outcsv)

    pickle.dump(g, open(outpklpath, 'wb'))

    plot_map(g, args.outdir)
    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()

