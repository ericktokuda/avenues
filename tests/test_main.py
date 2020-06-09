from py.test import approx
import os
import numpy as np
from src import main

def test_parse_graphml():

    try:
        main.parse_graphml('./data/test1.graphml', cachedir='/tmp/folder/',
                undir=True, samplerad=-1)
    except KeyError as e:
        print('Passed')

    try:
        main.parse_graphml('./data/test2.graphml', cachedir='/myfolder/',
                undir=True, samplerad=-1)
    except FileNotFoundError as e:
        print('Passed')

    cachedir = './cache/'

    g2 = main.parse_graphml('./data/test2.graphml', cachedir,
            undir=True, samplerad=-1)
    assert g2.is_connected() == True
    assert g2.is_directed() == False
    assert g2.is_simple() == True
    assert g2.vcount() == 2
    assert g2.ecount() == 1
    origs = np.where(np.array(g2.vs['type']) == main.ORIGINAL)[0]
    assert len(origs) == g2.vcount()
    origs = np.where(np.array(g2.es['type']) == main.ORIGINAL)[0]
    assert len(origs) == g2.ecount()
    assert os.path.exists(os.path.join(cachedir, 'test2.pkl'))
    diff = np.linalg.norm(np.array([0, 0]) - np.array([1, 1]))
    assert g2.es[0]['length'] == approx(diff)

    g3 = main.parse_graphml('./data/test3.graphml', cachedir,
            undir=True, samplerad=-1)
    assert g3.is_connected() == True
    assert g3.is_directed() == False
    assert g3.is_simple() == True
    assert g3.vcount() == 4
    assert g3.ecount() == 4
    origs = np.where(np.array(g3.vs['type']) == main.ORIGINAL)[0]
    assert len(origs) == g3.vcount()
    origs = np.where(np.array(g3.es['type']) == main.ORIGINAL)[0]
    assert len(origs) == g3.ecount()
    assert os.path.exists(os.path.join(cachedir, 'test3.pkl'))
    assert g3.es[0]['length'] == approx(0.001)

# def sample_circle_from_graph(g, radius):
# def get_points_inside_region(coords, c0, radius):
# def choose_bridge_endpoints(g, n):
# def calculate_edge_len(g, srcid, tgtid):
# def add_edge(g, srcid, tgtid, eid):
# def add_bridge_access(g, edge, coordstree, spacing, nnearest):
# def partition_edges(g, es, spacing, nnearest=1):
# def add_lengths(g):
# def calculate_avg_path_length(g, weighted=False, srctgttypes=None):
# def analyze_increment_of_random_edges(g, nnewedges, spacing, outcsv):
# def hex2rgb(hexcolours, normalized=False, alpha=None):
# def plot_map(g, outdir):

# def main():
