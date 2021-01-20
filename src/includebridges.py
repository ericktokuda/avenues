#!/usr/bin/env python3
"""Analysis of shortest paths in cities as we include shortcut connections.
We expect a network model or a graphml format with x,y attributes representing lon, lat
and we incrementally add bridges (or detour routes) and calculate the
shortest path lengths."""

import argparse
import time
import os, sys, random
from os.path import join as pjoin
import inspect

import numpy as np
from itertools import product
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc

import igraph
import pandas as pd
import scipy
from scipy.spatial import cKDTree
import itertools
from itertools import combinations
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl
from myutils import info, create_readme, append_to_file, graph, plot, geo
from optimized import generate_waxman_adj

#############################################################
ORIGINAL = 0
BRIDGE = 1
BRIDGEACC = 2
############################################################# BRIDGE LOCATION
UNIFORM = 0
DEGREE = 1
BETWV = 2
CLUCOEFF = 3
##########################################################
def parse_graphml(graphmlpath, undir=True, samplerad=-1):
    """Read graphml file to igraph object and dump it to @pklpath
    It gets the major component, and simplify it (neither multi nor self loops)
    Receives the input path @graphmlpath and dump to @pklpath.
    Assumes vertex attribs 'x' and 'y' are available """
    info(inspect.stack()[0][3] + '()')

    g = graph.simplify_graphml(graphmlpath, directed=False)
    g, origids = sample_circle_from_graph(g, samplerad)
    g = g.components(mode='weak').giant()
    g['vcount'] = g.vcount()
    g['ecount'] = g.ecount()

    g.vs['type'] = ORIGINAL
    g.es['type'] = ORIGINAL
    g.es['bridgeid'] = -1
    g.es['speed'] = 1
    g['coords'] = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
    return g, origids

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
    # info(inspect.stack()[0][3] + '()')

    if radius < 0: return g, list(range(g.vcount()))
    coords = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
    c0 = coords[np.random.randint(g.vcount())]
    ids = get_points_inside_region(coords, c0, radius)
    return induced_by(g, ids), ids

##########################################################
def get_points_inside_region(coords, c0, radius):
    """Get points from @df within circle of center @c0 and @radius"""
    # info(inspect.stack()[0][3] + '()')

    bt = BallTree(np.deg2rad(coords), metric='haversine')
    center = np.deg2rad(np.array([c0]))
    inds, dists = bt.query_radius(center, radius / geo.R,
            return_distance=True, sort_results=True)
    return inds[0]

##########################################################
def calculate_dist(coords, srcid, tgtid, realcoords):
    """Calculate edge length based on 'x' and 'y' attributes"""
    # src = np.array([float(g.vs[srcid]['x']), float(g.vs[srcid]['y'])])
    # tgt = np.array([float(g.vs[tgtid]['x']), float(g.vs[tgtid]['y'])])
    src = coords[srcid, :]
    tgt = coords[tgtid, :]
    if realcoords: l = geo.haversine(src, tgt)
    else: l = np.linalg.norm(tgt - src)
    return l

##########################################################
def add_wedge(g, srcid, tgtid, etype, bridgespeed, bridgeid=-1):
    g.add_edge(srcid, tgtid, type=etype)
    eid = g.ecount() - 1
    lon1, lat1 = float(g.vs[srcid]['x']), float(g.vs[srcid]['y'])
    lon2, lat2 = float(g.vs[tgtid]['x']), float(g.vs[tgtid]['y'])
    g.es[eid]['length'] = geo.haversine([lon1, lat1], [lon2, lat2])
    g.es[eid]['bridgeid'] = bridgeid
    g.es[eid]['speed'] = bridgespeed
    return g

##########################################################
def add_detour_route(g, edge, origtree, spacing, bridgeid, bridgespeed,
                     accessibs, nnearest):
    """Add shortcut path between @edge vertices"""
    info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0]
    coords = origtree.data
    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    vnorm = geo.haversine(src, tgt)
    versor = (tgt - src) / vnorm

    d = spacing

    lastpid = srcid
    vlast = srcid
    nnewedges = 0

    while d < vnorm:
        p = src + versor * d # in the worst case,
        _, ids = origtree.query(p, 3) # the 2 first may be the src and tgt

        for i, id in enumerate(ids):
            if orig[id] == srcid or orig[id] == tgtid: continue
            g = add_wedge(g, vlast, orig[id], BRIDGEACC, bridgespeed, bridgeid)
            g.vs[orig[id]]['type'] = BRIDGEACC
            break

        vlast = id
        d += spacing

    return add_wedge(g, vlast, tgtid, BRIDGEACC, bridgespeed, bridgeid)

##########################################################
def add_detour_route_accessibility(g, edge, origtree, spacing, bridgeid, bridgespeed,
                     accessibs, nnearest):
    """Add shortcut path between @edge vertices"""
    # info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0] # Real nodes
    coords = origtree.data
    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    vnorm = geo.haversine(src, tgt)
    versor = (tgt - src) / vnorm

    nemptyballs = 0

    d = spacing
    vlast = srcid
    while d < vnorm:
        p = src + versor * d # This vector increases with d
        ids = get_points_inside_region(coords, p, d/2)
        if len(ids) == 0: nemptyballs += 1
        ids = ids[np.argsort(accessibs[ids])] # sort by accessib

        wedgeadded = False
        for i, id in enumerate(ids):
            if not id in orig: continue
            elif id == srcid : continue
            elif id == tgtid: continue
            elif g.are_connected(vlast, id): continue

            g = add_wedge(g, vlast, id, BRIDGEACC, bridgespeed, bridgeid)
            g.vs[id]['type'] = BRIDGEACC
            wedgeadded = True
            break

        if wedgeadded: vlast = id
        d += spacing

    return add_wedge(g, vlast, tgtid, BRIDGEACC, bridgespeed, bridgeid), nemptyballs

##########################################################
def add_bridge(g, edge, origtree, spacing, bridgeid, nnearest, bridgespeed):
    """Add @eid bridge and accesses in @g"""
    # info(inspect.stack()[0][3] + '()')
    orig = np.where(np.array(g.vs['type']) != BRIDGEACC)[0]
    coords = origtree.data

    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    v = tgt - src
    vnorm = geo.haversine(src, tgt)
    versor = v / vnorm

    lastpid = srcid

    for d in np.arange(spacing, vnorm, spacing):
        p = src + versor * d
        params = {'type': BRIDGEACC, 'x': p[0], 'y': p[1]}
        g.add_vertex(p, **params) # new vertex in the bridge
        newvid = g.vcount() - 1
        nadded = len(np.where(np.array(g.vs['type']) == BRIDGEACC)[0])
        g.vs[newvid]['origid'] = g['vcount'] + nadded

        g = add_wedge(g, lastpid, newvid, BRIDGE, bridgespeed, bridgeid)
        g.vs[newvid]['x'] = p[0]
        g.vs[newvid]['y'] = p[1]
        _, ids = origtree.query(p, nnearest + 2)

        for i, id in enumerate(ids): # create accesses
            if i >= nnearest: break
            if orig[id] == srcid or orig[id] == tgtid: continue
            g = add_wedge(g, newvid, orig[id], BRIDGEACC, bridgespeed, bridgeid)

        lastpid = newvid

    g = add_wedge(g, lastpid, tgtid, BRIDGE, bridgespeed, bridgeid)
    return g

##########################################################
def partition_edges(g, endpoints, spacing, bridgeid, bridgespeed, accessibs,
                    nnearest=1):
    """Partition bridges spaced by @spacing and each new vertex is connected to
    the nearest node """
    origtree = cKDTree(g['coords'])
    # g = add_bridge(g, endpoints, origtree, spacing, bridgeid, nnearest,
                   # bridgespeed)
    # g = add_detour_route(g, endpoints, origtree, spacing, bridgeid,
                         # bridgespeed, accessibs, nnearest)
    g, nemptyballs = add_detour_route_accessibility(g, endpoints, origtree,
                                                    spacing, bridgeid,
                                                    bridgespeed, accessibs,
                                                    nnearest)

    return g, nemptyballs

##########################################################
def calculate_path_lengths(g, brspeed, weighted=False):
    """Calculate avg path length of @g.
    Calling all at once, without the loop on the vertices, it crashes
    for large graphs """
    # info(inspect.stack()[0][3] + '()')

    pathlens = {}

    if g.vcount() < 2: return 0, 0

    w = np.ones(g.ecount(), dtype=float)
    if weighted:
        bridgeids = np.where(np.array(g.es['type']) == BRIDGE)[0]
        if len(bridgeids) > 0:
            w[bridgeids] = np.array(g.es['length'])[bridgeids] / brspeed

    for srcid in range(g['vcount']): #Assuming an undirected graph
        tgts = list(range(srcid + 1, g['vcount']))

        spaths =  g.shortest_paths(source=srcid, target=tgts,
                                   weights=w, mode='ALL')

        for tgtid, l in zip(tgts, spaths[0]):
            pathlens[(srcid, tgtid)] = l

    return pathlens

##########################################################
def extract_features(g, bridgespeed):
    """Extract features from graph @g """
    # info(inspect.stack()[0][3] + '()')
    degrees = np.array(g.degree())
    pathlens = calculate_path_lengths(g, bridgespeed, weighted=True)
    betwv = np.array(g.betweenness())
    clucoeff = np.array(g.transitivity_local_undirected(mode="nan"))
    divers = np.array(g.diversity(weights=g.es['length']))
    clos = np.array(g.closeness())

    pathlensv = np.array(list(pathlens.values()))
    assort = g.assortativity(g.degree(), directed=False)
    clucoeff_ = clucoeff[~np.isnan(clucoeff)]
    divers_ = divers[~np.isnan(divers)]
    divers_ = divers_[np.isfinite(divers_)]

    features = dict(
            g_pathlen_mean = np.mean(pathlensv),
            g_pathlen_std = np.std(pathlensv),
            g_degree_mean = np.mean(degrees),
            g_degree_std = np.std(degrees),
            g_betwv_mean = np.mean(betwv),
            g_betwv_std = np.std(betwv),
            g_assort_mean = np.mean(assort),
            g_assort_std = np.std(assort),
            g_clucoeff_mean = np.mean(clucoeff_),
            g_clucoeff_std = np.std(clucoeff_),
            g_divers_mean = np.mean(divers_),
            g_divers_std = np.std(divers_),
            g_clos_mean = np.mean(clos),
            g_clos_std = np.std(clos),
            )

    features['nbridges'] = len(np.unique(g.es['bridgeid'])) - 1
    features['naccess'] = len(np.where(np.array(g.es['type']) == BRIDGEACC)[0])
    return features

##########################################################
def analyze_increment_of_bridges(gorig, bridges, ndetours, bridgespeed, accessibs,
                                 outdir, outcsv):
    """Analyze increment of @bridges to @g. We add entrances/exit spaced
    by @spacing and output to @outcsv."""
    info(inspect.stack()[0][3] + '()')

    nbridges = len(bridges)

    feats = extract_features(gorig, bridgespeed)
    vals = [feats.values()]

    sumemptyballs = 0
    g = gorig.copy()
    for bridgeid, es in enumerate(bridges):
        info('bridgeid:{}'.format(bridgeid))
        src, tgt = es
        g.vs[src]['type'] = g.vs[tgt]['type'] = BRIDGE
        d = geo.haversine(g['coords'][src, :], g['coords'][tgt, :])
        spacing = d / ndetours + .001
        g, nemptyballs = partition_edges(g, es, spacing, bridgeid, bridgespeed,
                                         accessibs, nnearest=1)
        sumemptyballs += nemptyballs
        vtypes = np.array(g.vs['type'])
        vals.append(extract_features(g, bridgespeed).values())
        # plot_map(g, pjoin(outdir, 'map_{:02d}.png'.format(bridgeid)),
        # vertices=True)

    outpath = pjoin(outdir, 'finalmap.pdf')
    plot_map(g, outpath, vertices=True)
    df = pd.DataFrame(vals, columns=feats.keys())
    df.to_csv(outcsv, index=False)
    return sumemptyballs

##########################################################
def plot_map(g, outpath, vertices=False):
    """Plot map g, according to 'type' attrib both in vertices and in edges """

    info(inspect.stack()[0][3] + '()')
    vtypes = np.array(g.vs['type'])
    etypes = np.array(g.es['type'])

    nrows = 1;  ncols = 1
    figscale = 10
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

    lc = mc.LineCollection(lines, colors=ecolours, linewidths=figscale*.1,
                           zorder=0)
    axs[0, 0].add_collection(lc)
    axs[0, 0].autoscale()

    if vertices:
        vids = np.where(np.array(g.vs['type']) == ORIGINAL)[0]
        axs[0, 0].scatter(g['coords'][vids, 0], g['coords'][vids, 1],
                          s=figscale*.05, c=[palettergb[0]], zorder=5)

        vids = np.where(np.array(g.vs['type']) == BRIDGEACC)[0]
        axs[0, 0].scatter(g['coords'][vids, 0], g['coords'][vids, 1],
                          s=figscale*.1, c=[palettergb[1]], zorder=10)

    vids = np.where(np.array(g.vs['type']) == BRIDGE)[0]
    axs[0, 0].scatter(g['coords'][vids, 0], g['coords'][vids, 1],
                      s=figscale*.1, c='k', zorder=7)
    # axs[0, 0].set_xticks([])
    # axs[0, 0].set_yticks([])

    plt.savefig(outpath)

##########################################################
def choose_bridges_random_minlen(g, nnewedges, available, minlen):
    inds = np.arange(len(available))
    np.random.shuffle(inds)

    bridges = []
    for ind in inds:
        srcid, tgtid = available[ind]
        l = calculate_dist(g['coords'], srcid, tgtid, g['isreal'])
        if l > minlen: bridges.append(available[ind])
        if len(bridges) == nnewedges: break

    return bridges

##########################################################
def weighted_random_sampling_n(items, weights, n):
    item = items.copy(); weights = weights.copy()
    sample = np.zeros(n, dtype=int)
    inds = list(range(len(items)))

    for i in range(n):
        sampleidx = weighted_random_sampling(items, weights, return_idx=True)
        sample[i] = items[sampleidx]
        items = np.delete(items, sampleidx)
        weights = np.delete(weights, sampleidx)

    return sample

##########################################################
def choose_new_bridges(g, nnewedges, length, choice, eps=0):
    """Choose new edges. Multiple edges are not allowed.
    We compute the set difference between all possible edges
    and the existing ones."""
    info(inspect.stack()[0][3] + '()')

    # All possible edges
    l1 = (1 - eps) * length
    l2 = (1 + eps) * length
    coords = g['coords']
    # dists = euclidean_distances(coords, coords)
    combs, dists = geo.haversine_all(coords)
    withinrange = ((dists > l1) & (dists < l2))
    combs = combs[withinrange]
    inds = (combs[:, 0], combs[:, 1])
    adj1 = np.zeros((g.vcount(), g.vcount()), dtype=int)
    adj1[inds] = 1
    adj2 = (np.array(g.get_adjacency().data)).astype(bool)

    available = np.logical_and(adj1, ~adj2)
    available = np.where(available == True)
    m = len(available[0])

    if choice == UNIFORM:
        sampleids = list(range(m))
        np.random.shuffle(sampleids)
        sampleids = sampleids[:nnewedges]
    elif choice == DEGREE:
        # TODO: preference of each vertex or of both
        weights = np.array(g.degree())
        sampleids = weighted_random_sampling_n(list(range(g.vcount())),
                weights, nnewedges)

    return np.array([ available[0][sampleids], available[1][sampleids]]).T

    # return choose_bridges_random_minlen(g, nnewedges, available, minlen)
    # return choose_bridges_given_len(g, nnewedges, available, length)

##########################################################
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

#############################################################
def get_waxman_params(nvertices, avgdegree, alpha):
    wxcatalog = {
         '1000,6,0.0050': 179.3795937768998,
        '11132,6,0.0010': 3873.8114499435333,
        '11132,6,0.0050': 1.8043511010431004,
        '11132,6,0.0100': 0.4426419682443505,
    }

    if '{},{},{:.04f}'.format(nvertices, avgdegree, alpha) in wxcatalog.keys():
        return wxcatalog['{},{},{:.04f}'.format(nvertices, avgdegree, alpha)], alpha

    maxnedges = nvertices * nvertices // 2
    def f(b):
        g = generate_waxman(nvertices, maxnedges, alpha=alpha, beta=b)
        return np.mean(g.degree()) - avgdegree

    beta = scipy.optimize.brentq(f, 0.0001, 10000, xtol=0.00001, rtol=0.01)
    info('beta:{}'.format(beta))
    return beta, alpha

#############################################################
def generate_waxman(n, maxnedges, alpha, beta, domain=(0, 0, 1, 1)):
    adjlist, x, y = generate_waxman_adj(n, maxnedges, alpha, beta,
                                        domain[0], domain[1], domain[2], domain[3])
    adjlist = adjlist.astype(int).tolist()

    g = igraph.Graph(n, adjlist)
    g.vs['x'] = x
    g.vs['y'] = y
    return g

#############################################################
def generate_graph(topologymodel, nvertices, avgdegree, wxalpha):
    info(inspect.stack()[0][3] + '()')
    if topologymodel == 'gr':
        radius = get_rgg_params(nvertices, avgdegree)
        g = igraph.Graph.GRG(nvertices, radius)
    elif topologymodel == 'wx':
        beta, alpha = get_waxman_params(nvertices, avgdegree, wxalpha)
        maxnedges = nvertices * nvertices // 2
        g = generate_waxman(nvertices, maxnedges, beta=beta, alpha=alpha)

    g['isreal'] = False
    g = g.clusters().giant()

    aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
    g['vcount'] = g.vcount()
    g['ecount'] = g.ecount()

    g.vs['type'] = ORIGINAL
    g.es['type'] = ORIGINAL
    g.es['bridgeid'] = -1
    g.es['speed'] = 1

    g['coords'] = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0))

    for j, e in enumerate(g.es()):
        l = calculate_dist(g['coords'], e.source, e.target, g['isreal'])
        g.es[j]['length'] = l
    return g

##########################################################
def get_rgg_params(nvertices, avgdegree):
    rggcatalog = {
        '625,6': 0.056865545,
        '10000,6': 0.0139,
        '11132,6': 0.0131495,
        '22500,6': 0.00925,
    }

    if '{},{}'.format(nvertices, avgdegree) in rggcatalog.keys():
        return rggcatalog['{},{}'.format(nvertices, avgdegree)]

    def f(r):
        g = igraph.Graph.GRG(nvertices, r)
        return np.mean(g.degree()) - avgdegree

    return scipy.optimize.brentq(f, 0.0001, 10000)

##########################################################
def plot_topology(g, coords, toprasterpath, visualorig, plotalpha):
    """Plot the gradients map"""
    info(inspect.stack()[0][3] + '()')
    visual = visualorig.copy()
    visual['vertex_size'] = 0
    gradientscolors = [1, 1, 1]
    igraph.plot(g, target=toprasterpath, layout=coords.tolist(),
                 **visual)

##########################################################
def define_plot_layout(mapside, plotzoom):
    # Square of the center surrounded by radius 3
    #  (equiv to 99.7% of the points of a gaussian)
    visual = dict(
        bbox = (mapside*10*plotzoom, mapside*10*plotzoom),
        margin = mapside*plotzoom,
        vertex_size = 5*plotzoom,
        vertex_shape = 'circle',
        # vertex_frame_width = 0
        vertex_frame_width = 0.1*plotzoom,
        edge_width=1.0
    )
    return visual

##########################################################
def scale_coords(g, bbox):
    """Scale [-1,1] coords to the bbox provided"""
    info(inspect.stack()[0][3] + '()')
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax - xmin) / 2
    dy = (ymax - ymin) / 2
    delta = np.array([dx, dy])
    c0 = [xmin + dx, ymin + dy]
    coords = c0 + (delta * g['coords'])
    g['coords'] = coords
    g.vs['x'] = coords[:, 0]
    g.vs['y'] = coords[:, 1]

    for j, e in enumerate(g.es()):
        l = calculate_dist(g['coords'], e.source, e.target, g['isreal'])
        g.es[j]['length'] = l
    return g

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True,
            help='Path to the graphml OR wx OR gr')
    parser.add_argument('--accessibpath',
            help='Path to the accessib values')
    parser.add_argument('--wxalpha', default=.1, type=float,
            help='Waxman alpha')
    parser.add_argument('--bridgelen', default=.5, type=float,
            help='Length of the bridges relative to the avg. path length')
    parser.add_argument('--nbridges', default=3, type=int,
            help='Number of bridges')
    parser.add_argument('--bridgespeed', default=1.0, type=float,
            help='Speed in bridges')
    parser.add_argument('--samplerad', default=-1, type=float,
            help='Region of interest radius')
    parser.add_argument('--seed', default=0, type=int,
            help='Randomicity seed')
    parser.add_argument('--outdir', default='/tmp/out/',
            help='Output directory')
    args = parser.parse_args()

    nvertices = 11132 #11132 is the mean of the 4 cities
    avgdegree = 6

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    outcsv = pjoin(args.outdir, 'results.csv')
    maxnedges = np.max(args.nbridges)

    if os.path.exists(args.graph):
        g, origids = parse_graphml(args.graph, undir=True, samplerad=args.samplerad)
        g['isreal'] = True
    elif args.graph in ['wx', 'gr']:
        g = generate_graph(args.graph, nvertices, avgdegree, args.wxalpha)
    else:
        info('Please provide a proper graph argument.')
        info('Either a graphml path OR waxman OR geometric')
        return

    if args.accessibpath:
        with open(args.accessibpath) as fh:
            accessibs = np.array([float(x) for x in fh.read().strip().split('\n')])
            accessibs = accessibs[origids]
    else:
        accessibs = np.ones(g.vcount(), dtype=float)

    diam = g.diameter(weights='length')
    visual = define_plot_layout(100, 1)
    plot_topology(g, g['coords'], pjoin(args.outdir, 'graph.pdf'), visual, .5)

    info('nvertices: {}'.format(g.vcount()))
    info('nedges: {}'.format(g.ecount()))

    ndetours = 5

    append_to_file(readmepath, 'vcount:{},ecount:{}'.format(g.vcount(), g.ecount()))
    append_to_file(readmepath, 'diameter:{},bridgelen:{},ndetours:{}'.format(
        diam, args.bridgelen, ndetours))

    eps = .1 # 10% of margin
    es = choose_new_bridges(g, args.nbridges, args.bridgelen, UNIFORM, eps=.1)

    sumemptyballs = analyze_increment_of_bridges(g, es, ndetours, args.bridgespeed,
                                                 accessibs, args.outdir, outcsv)
    append_to_file(readmepath, 'sumemptyballs:{}'.format(sumemptyballs))

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
