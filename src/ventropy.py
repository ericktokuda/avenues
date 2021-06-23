#!/usr/bin/env python3
"""Estimate entropy of the rastered graph  considering the node positions
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
import osmnx as ox
import scipy.ndimage as nd

from myutils import info, create_readme

##########################################################
def get_pos_array(graph, latlon=False):
    """Get numpy array containing the positions of each node in a projected
    osmnx network.  If latlon==True, use latidude and longitude coordinates.
    Otherwise, the projected positions are used. Note that if the plot has not
    been projected, latlon==True will give an error and latlon==False
    will return the latitude and longitude."""

    nodes = list(graph.nodes(data=True))
    if latlon:
        y = [float(node[1]['lat']) for node in nodes]
        x = [float(node[1]['lon']) for node in nodes]
    else:
        y = [node[1]['y'] for node in nodes]
        x = [node[1]['x'] for node in nodes]
    pos_nodes = np.array([x,y]).T
    return pos_nodes

##########################################################
def calculate_rastered_graph_entropy(graphmlpath, sigma, min_width, bin_size, pad=1000):
    '''Calculate the entropy of the input graph. Note that the function expects
    variable graph to contain two atributes named 'x' and 'y' associated to
    the spatial position of the nodes.  '''

    graph = ox.load_graphml(graphmlpath)
    graph = ox.project_graph(graph)
    pos = get_pos_array(graph)
    posx, posy = pos.T

    xmin,xmax = np.min(posx),np.max(posx)
    ymin,ymax = np.min(posy),np.max(posy)

    sigma_bins = sigma/bin_size
    min_width_bins = min_width/bin_size

    binsC = np.arange(xmin-pad, xmax+pad+0.5*bin_size, bin_size)
    binsR = np.arange(ymin-pad, ymax+pad+0.5*bin_size, bin_size)

    info('Using %d and %d bins on x and y directions'%(len(binsC), len(binsR)))

    hist, _, _ = np.histogram2d(posy, posx,[binsR, binsC])
    distr = nd.gaussian_filter(hist, sigma_bins)

    distr = distr / np.sum(distr)
    distr = distr[distr > 0]
    return -(distr*np.log(np.abs(distr))).sum()

##########################################################
def main():
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    info('For Aiur!')

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

##########################################################
if __name__ == "__main__":
    main()
