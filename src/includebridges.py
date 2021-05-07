#!/usr/bin/env python3
"""Analysis of shortest paths in cities as we include shortcut connections.
We expect a network model or a graphml format with x,y attributes representing lon, lat
and we incrementally add bridges (or detour routes) and calculate the
shortest path lengths."""

import time, os, sys, math, random
from optimized import generate_waxman_adj
from myutils import info, create_readme, append_to_file, graph, plot, geo
import pickle as pkl
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import BallTree
from itertools import combinations
import itertools
from scipy.spatial import cKDTree
import scipy
import pandas as pd
import igraph
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import argparse
from os.path import join as pjoin
import inspect
from multiprocessing import Pool

import numpy as np
from itertools import product
import matplotlib
matplotlib.use('Agg')

#############################################################
ORIGINAL = 0
BRIDGE = 1
BRIDGEACC = 2

# BRIDGE LOCATION
UNIFORM = 0
DEGREE = 1
BETWV = 2
CLUCOEFF = 3

##########################################################
def parse_graphml(graphmlpath, undir=True, samplerad=-1):
    """Read graphml file to igraph object and dump it to @pklpath
    It extracts the major component, and simplify it (neither multiedges nor
    self loops). It expects the input path @graphmlpath and dumps to @pklpath.
    It assumes vertex attribs 'x' and 'y' are available."""
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

    if radius < 0:
        return g, list(range(g.vcount()))
    coords = np.array([(x, y) for x, y in zip(g.vs['x'], g.vs['y'])])
    c0 = coords[np.random.randint(g.vcount())]
    ids = get_points_inside_region(coords, c0, radius)
    return induced_by(g, ids), ids

##########################################################
def get_points_inside_region(coords, c0, radius):
    """Get points from @df within circle of center @c0 and @radius"""

    bt = BallTree(np.deg2rad(coords), metric='haversine')
    center = np.deg2rad(np.array([c0]))
    inds, dists = bt.query_radius(center, radius / geo.R,
                                  return_distance=True, sort_results=True)
    return inds[0]

##########################################################
def calculate_dist(coords, srcid, tgtid, realcoords):
    """Calculate edge length based on 'x' and 'y' attributes"""
    src = coords[srcid, :]
    tgt = coords[tgtid, :]
    if realcoords:
        l = geo.haversine(src, tgt)
    else:
        l = np.linalg.norm(tgt - src)
    return l

##########################################################
def add_wedge(g, srcid, tgtid, etype, bridgespeed, bridgeid=-1):
    """Add weighted edge to @g"""
    g.add_edge(srcid, tgtid, type=etype)
    eid = g.ecount() - 1
    lon1, lat1 = float(g.vs[srcid]['x']), float(g.vs[srcid]['y'])
    lon2, lat2 = float(g.vs[tgtid]['x']), float(g.vs[tgtid]['y'])
    g.es[eid]['length'] = geo.haversine([lon1, lat1], [lon2, lat2])
    g.es[eid]['bridgeid'] = bridgeid
    g.es[eid]['speed'] = bridgespeed
    return g

##########################################################
def add_path_closest(g, bridgeid, es, bridgespacing, bridgespeed):
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
        p = src + versor * d  # in the worst case,
        _, ids = origtree.query(p, 3)  # the 2 first may be the src and tgt

        for i, id in enumerate(ids):
            if orig[id] == srcid or orig[id] == tgtid:
                continue
            g = add_wedge(g, vlast, orig[id], BRIDGEACC, bridgespeed, bridgeid)
            g.vs[orig[id]]['type'] = BRIDGEACC
            break

        vlast = id
        d += spacing

    return add_wedge(g, vlast, tgtid, BRIDGEACC, bridgespeed, bridgeid)

##########################################################
def add_avenue_closest(g, bridgeid, edge, bridgespacing, bridgespeed, coordstree):
    """Add a path between the @edge vertices with points in-between
    spaced by @bridgespacing and with speed @bridgespeed.
    The detour vertices are chosen based on the minimum
    distances to the the straight line middle points.  """
    return add_avenue(g, bridgeid, edge, bridgespacing, bridgespeed,
                      'uniform', coordstree)

##########################################################
def add_avenue_accessib(g, bridgeid, edge, bridgespacing, bridgespeed, accessibs):
    """Add a path between the @edge vertices with points in-between
    spaced by @bridgespacing and with speed @bridgespeed.
    The detour vertices are chosen based on the accessibility measurement
    close to the stright line middle points.  """
    return add_avenue(g, bridgeid, edge, bridgespacing, bridgespeed,
                      'accessib', accessibs)

##########################################################
def add_avenue(g, bridgeid, edge, bridgespacing, bridgespeed, choice, choiceparam):
    """Add a path between @edge vertices with @bridgespacing in between and with
    speed @bridgespeed."""
    coords = g['coords']
    srcid, tgtid = edge
    src = coords[srcid]
    tgt = coords[tgtid]
    vnorm = geo.haversine(src, tgt)
    versor = (tgt - src) / vnorm

    ndetours = np.round(vnorm / bridgespacing).astype(int) - 1
    if ndetours > 0: d = (vnorm / ndetours)

    avvertices = [srcid]

    for j in range(ndetours):
        i = j + 1  # 1-based index
        p = src + versor * (d * i)  # This vector p increases with d

        if choice == 'uniform':
            # the first 2 may be the src and tgt
            _, ids = choiceparam.query(p, 4)
        elif choice == 'accessib':
            ids = get_points_inside_region(coords, p, d/2)
            ids = ids[np.argsort(choiceparam[ids])]  # Sort by accessib
            ids = list(reversed(ids))

        emptyball = True
        for i, id in enumerate(ids):
            if id == avvertices[-1] or id == srcid or id == tgtid:
                continue

            if not g.are_connected(avvertices[-1], id):  # Avoid multi-edges
                g = add_wedge(g, avvertices[-1], id, BRIDGE, bridgespeed, bridgeid)

            g.vs[id]['type'] = BRIDGEACC
            emptyball = False
            break

        if emptyball: # In case we cannot not find any vertex nearby
            return g, False

        avvertices.append(id)

    # Last segment of the bridge
    if not g.are_connected(avvertices[-1], tgtid):
        g = add_wedge(g, avvertices[-1], tgtid, BRIDGE, bridgespeed, bridgeid)

    avvertices.append(tgtid)
    return g, avvertices, True

##########################################################
def calculate_path_lengths(g, brspeed, weighted=False):
    """Calculate avg path length of @g.
    Calling all at once, without the loop on the vertices, it crashes
    for large graphs """
    # info(inspect.stack()[0][3] + '()')

    pathlens = {}

    n = g.vcount()
    if n < 2:
        return 0, 0

    w = np.array(g.es['length'])
    if weighted:
        bridgeids = np.where(np.array(g.es['type']) == BRIDGE)[0]
        if len(bridgeids) > 0:
            w[bridgeids] = w[bridgeids] / brspeed

    paths = np.array(g.shortest_paths(weights=w))
    xu, yu = np.triu_indices_from(paths, k=1)  # Remove diagonal and duplicates
    return paths[xu, yu]

##########################################################
def analyze_increment_of_bridge(params):
    """Add one bridge and extract features"""
    bridgeid, gorig, edge, bridgespacing, bridgespeed, coordstree = params
    info('bridge {}'.format(bridgeid))
    g = gorig.copy()
    orignedges = g.ecount()

    g, avvertices, succ = add_avenue_closest(g, bridgeid, edge, bridgespacing,
                                             bridgespeed, coordstree)

    if not succ: return [-1], [-1] * 16

    g.vs[edge[0]]['type'] = g.vs[edge[1]]['type'] = BRIDGE
    plen = calculate_path_lengths(g, bridgespeed, weighted=True)
    return avvertices, np.mean(plen)

##########################################################
def analyze_increment_of_bridges(gorig, bridges, bridgespacing,
                                 bridgespeed, nprocs, outdir):
    """Increment of @bridges to @g and extract features from each state. We add
    entrances/exit separated by @bridgespacing (+eps).
    The new edges have different speeds (@bridgespeed) in the shortest
    paths computation."""
    info(inspect.stack()[0][3] + '()')

    plen0 = calculate_path_lengths(gorig, bridgespeed, weighted=True)
    avgplen0 = np.mean(plen0)

    coordstree = cKDTree(gorig['coords'])

    aux = product([gorig], bridges, [bridgespacing],
                  [bridgespeed], [coordstree],)
    aux = [list(a) for a in list(aux)] # Convert to list
    [a.insert(0, i) for i, a in enumerate(list(aux))] # Add counter

    if nprocs == 1:
        info('Running serially (nprocs:{})'.format(nprocs))
        ret = [analyze_increment_of_bridge(p) for p in aux]
    else:
        info('Running in parallel (nprocs:{})'.format(nprocs))
        pool = Pool(nprocs)
        ret = pool.map(analyze_increment_of_bridge, aux)

    avvertices = [r[0] for r in ret]
    avgplens = [r[1] for r in ret]

    pkl.dump(avvertices, open(pjoin(outdir, 'avenues.pkl'), 'wb'))
    with open(pjoin(outdir, 'avenues.txt'), 'w') as fh:
        for av in avvertices: fh.write(','.join(str(v) for v in av)); fh.write('\n')

    return avgplen0 - avgplens

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
    """Generate graph according to the @topologymodel, @nvertices, @avgdegree
    and @wxalpha, if applicable. It also compute and store the edges lengths
    in the 'length' attribute."""
    info(inspect.stack()[0][3] + '()')
    if topologymodel == 'gr':
        radius = get_rgg_params(nvertices, avgdegree)
        g = igraph.Graph.GRG(nvertices, radius)
    elif topologymodel == 'wx':
        beta, alpha = get_waxman_params(nvertices, avgdegree, wxalpha)
        maxnedges = nvertices * nvertices // 2
        g = generate_waxman(nvertices, maxnedges, beta=beta, alpha=alpha)

    g['isreal'] = False
    g['topology'] = topologymodel
    g = g.clusters().giant()

    aux = np.array([[g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount())])
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
        bbox=(mapside*10*plotzoom, mapside*10*plotzoom),
        margin=mapside*plotzoom,
        vertex_size=5*plotzoom,
        vertex_shape='circle',
        # vertex_frame_width = 0
        vertex_frame_width=0.1*plotzoom,
        edge_width=1.0
    )
    return visual

##########################################################
def get_dcoords(angrad, bridgelen, refcoords):
    """x is vertical and y is horizontal"""
    dcoords = np.ndarray((len(angrad), 2), dtype=float)
    for i, b in enumerate(angrad):
        a = b % (2 * np.pi) # Remove loops
        versorx = math.sin(a)
        versory = math.cos(a)
        versor = np.array([versorx, versory])

        def get_coords_delta(x, midpoint=refcoords, versor=versor):
            h = geo.haversine(refcoords, refcoords + x * versor)
            return bridgelen - h

        d = scipy.optimize.bisect(get_coords_delta, 0, 0.1,
                                  xtol=0.001, rtol=0.01)
        dcoords[i, :] = d * versor
    return dcoords

##########################################################
def load_graph(graph, samplerad):
    if os.path.exists(graph): # Load Graph
        g, origids = parse_graphml(
            graph, undir=True, samplerad=samplerad)
        g['isreal'] = True
        g['topology'] = os.path.basename(graph)
    elif graph in ['wx', 'gr']:
        g = generate_graph(graph, refvcount, avgdegree, wxalpha)
    else:
        info('Please provide a proper graph argument.')
        info('Either a graphml path OR waxman OR geometric')
        return None
    return g

##########################################################
def calculate_bridge_edpoints(coords, bounds, gridx, gridy,
                              angrad, bridgelen, maxdist, outdir):
    coordstree = cKDTree(coords)
    refcoords = np.array([(bounds[2] - bounds[0]) / 2,
                          (bounds[3] - bounds[1]) / 2])

    # Calculate the (dx, dy) for each angle
    dcoords = get_dcoords(angrad, bridgelen, refcoords)

    gridlen = gridx.shape[0] * gridx.shape[1] * len(angrad)
    esexact = np.zeros((gridlen, 4), dtype=float)
    gridvalidids = [] # Keep the id of the valid grid
    es = []
    i = 0
    for x, y in zip(gridx.flatten(), gridy.flatten()):
        grid0 = np.array([x, y])
        _, src = coordstree.query(grid0)
        d1 = geo.haversine(grid0, coords[src])
        for j, a in enumerate(angrad):
            p = grid0 + dcoords[j, :]
            esexact[i, :2] = grid0; esexact[i, 2:] = p
            _, outtgt = coordstree.query(p, k=2)
            tgt = outtgt[1] if outtgt[0] == src else outtgt[0]

            # d = geo.haversine(coords[src], coords[tgt])
            d2 = geo.haversine(p, coords[tgt])
            if d1 <= maxdist and d2 <= maxdist:
                es.append([src, tgt])
                gridvalidids.append(i)

            i += 1

    # Store bridge endpoints (grid and real)
    pkl.dump(esexact, open(pjoin(outdir, 'brcoordsexact.pkl'), 'wb'))

    es = np.array(es)
    brcoords = np.concatenate([coords[es[:, 0]],coords[es[:, 1]]], axis=1)
    pkl.dump(brcoords, open(pjoin(outdir, 'brcoords.pkl'), 'wb'))
    return es, esexact, brcoords, gridvalidids

##########################################################
def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True,
                        help='Path to the graphml OR wx OR gr')
    parser.add_argument('--accessibpath',
                        help='Path to the accessib values')
    parser.add_argument('--wxalpha', default=.1, type=float,
                        help='Waxman alpha')
    parser.add_argument('--nbridgeangles', default=4, type=int,
                        help='Number of bridge angles')
    parser.add_argument('--bridgespacing', default=0.5, type=float,
                        help='Spacing between the middle points of the bridge.')
    parser.add_argument('--bridgespeed', default=1.0, type=float,
                        help='Speed in bridges')
    parser.add_argument('--gridside', default=4, type=int,
                        help='Side of the grid')
    parser.add_argument('--samplerad', default=-1, type=float,
                        help='Region of interest radius')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--nprocs', default=1, type=int,
                        help='Number of processes (parallel)')
    parser.add_argument('--outdir', default='/tmp/out/',
                        help='Output directory')
    return parser.parse_args()

##########################################################
def export_results(nbridges, graph, bridgespeed, brcoordsexact,
                   brcoords, gridvalidids, gains, ndigits, outdir):
    """Export results to csv file"""
    city = os.path.basename(graph).replace('.graphml', '')
    col1 = np.array([city]* nbridges).reshape(-1, 1)
    col2 = np.array([bridgespeed]* len(gains)).reshape(-1, 1)
    gains = gains.reshape(-1, 1)

    delta = brcoordsexact[:, :2] - brcoordsexact[:, 2:]
    brexactangle = np.arctan2(delta[:, 0], delta[:, 1]).reshape(-1, 1)

    delta = brcoords[:, :2] - brcoords[:, 2:]
    brangle = np.arctan2(delta[:, 0], delta[:, 1]).reshape(-1, 1)

    brcoordsexact = np.around(brcoordsexact[gridvalidids], ndigits)
    brexactangle = np.around(brexactangle[gridvalidids], ndigits)
    brcoords = np.around(brcoords, ndigits)
    brangle = np.around(brangle, ndigits)
    # gains = np.around(gains, ndigits) # Gains are generally very small
    gridvalidids = np.array(gridvalidids).reshape(-1, 1)

    x = np.concatenate([col1, col2, brcoordsexact, brexactangle,
                        gridvalidids, brcoords, brangle, gains], axis=1)

    cols = ('city,brspeed,' \
        'brexactsrcx,brexactsrcy,brexacttgtx,brexacttgty,brexactangle,' \
        'gridid,brsrcx,brsrcy,brtgtx,brtgty,brangle,' \
        'gain').split(',')

    outpath = pjoin(outdir, 'results.csv')
    pd.DataFrame(x, columns=cols).to_csv(outpath, index=False,
                                         float_format='%.4f')
    return

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    refvcount = 11132  # Mean of the 4 cities
    avgdegree = 6
    bridgeleneps = .1  # 10% of margin
    bridgeprop = .25 # Proportion wrt the graph diameter
    ndigits = 4

    random.seed(args.seed); np.random.seed(args.seed)

    g = load_graph(args.graph, args.samplerad)
    if g == None: return

    visual = define_plot_layout(100, 1)
    plot_topology(g, g['coords'], pjoin(args.outdir, 'graph.pdf'), visual, .5)

    diam = g.diameter(weights='length')

    xmin, ymin = np.min(g['coords'], axis=0)
    xmax, ymax = np.max(g['coords'], axis=0)
    bounds = np.array([xmin, ymin, xmax, ymax])
    bridgelen = bridgeprop * diam

    gridx, gridy = np.mgrid[bounds[0]:bounds[2]:(args.gridside*1j),
                            bounds[1]:bounds[3]:(args.gridside*1j)]
    angrad = np.linspace(-np.pi, np.pi, args.nbridgeangles, endpoint=False)

    # Euclidean distance
    p0 = np.array([gridx[0, 0], gridy[0, 0]])
    p1 = np.array([gridx[1, 1], gridy[1, 1]])
    # griddist2 = (gridx[1, 0] - gridx[0, 0]) ** 2 + (gridy[0, 1] - gridy[0, 0]) ** 2
    maxdist = geo.haversine(p0, p1) / 2

    r = calculate_bridge_edpoints(g['coords'], bounds, gridx, gridy,
                                  angrad, bridgelen, maxdist,
                                  args.outdir)
    es, brcoordsexact, brcoords, gridvalidids = r
    nbridges = len(gridvalidids)

    # Bulk of the processing
    gains = analyze_increment_of_bridges(g, es, args.bridgespacing,
                                         args.bridgespeed, args.nprocs,
                                         args.outdir)

    export_results(nbridges, args.graph, args.bridgespeed, brcoordsexact,
                   brcoords, gridvalidids, gains, ndigits, args.outdir)

    s = 'vcount:{},ecount:{},diameter:{:.04f},bridgelen:{:.04f},' \
        'bridgespacing:{:.04f},maxdist:{:.04f}\n' . \
        format(g.vcount(), g.ecount(), diam, bridgelen, args.bridgespacing,
               maxdist)
    append_to_file(readmepath, s.replace(',', '\n'))

    info('Elapsed time:{}'.format(time.time()-t0))

##########################################################
if __name__ == "__main__":
    main()
