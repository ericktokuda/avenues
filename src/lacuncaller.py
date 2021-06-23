#!/usr/bin/env python3
"""Calculate lacunarity
"""
import numpy as np
import networkx as nx
from skimage.draw import line

import argparse
import time
import os
from os.path import join as pjoin
import inspect

import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from myutils import info, create_readme
from lacunarity import lacunarity

##########################################################
def haversine(lat, lon, lat_vet, lon_vet):
    """
    Calculate the great circle distance between a point and a point or
    a set of points on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon = np.radians(lon)
    lat = np.radians(lat)
    lon_vet = np.radians(lon_vet)
    lat_vet = np.radians(lat_vet)

    # haversine formula 
    dlon = lon_vet - lon 
    dlat = lat_vet - lat 
    a = np.sin(dlat/2.)**2 + np.cos(lat) * np.cos(lat_vet) * np.sin(dlon/2.)**2
    c = 2. * np.arcsin(np.sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367. * c
    return km   

def get_attribute_as_array(graph, att_name):
    return np.array([float(v) for k, v in graph.nodes(data=att_name)])

def draw_city(city_graph, px_per_km=100):
    """Draw city graph as a rasterized image. `city_graph` should be a networkx graph
    containing attributes 'posx' (longitude) and 'posy' (latitude). `px_per_km` defines the 
    number of pixels used to represent each kilometer."""

    # Get latitude and longitude of the nodes
    lon = get_attribute_as_array(city_graph, 'x')
    lat = get_attribute_as_array(city_graph, 'y')

    min_lon, min_lat = np.min(lon), np.min(lat)

    # Transform latitude and longitude to kilometers
    pos_x = haversine(lat, min_lon, lat, lon)
    pos_y = haversine(min_lat, lon, lat, lon)

    pos = np.array([pos_x, pos_y]).T    
    max_x, max_y = np.max(pos, axis=0)

    size_x = int(np.round(max_x*px_per_km)) + 1
    size_y = int(np.round(max_y*px_per_km)) + 1

    # Get coordinates of the nodes in pixels
    pos_img = np.zeros_like(pos, dtype=int)
    pos_img[:,0] = (np.round(pos[:,0]*px_per_km)).astype(int) + 1
    pos_img[:,1] = size_y - (np.round(pos[:,1]*px_per_km)).astype(int)

    # Map of nodes names to integers
    node_index_map = dict(zip(city_graph.nodes, range(city_graph.number_of_nodes())))

    # Draw lines between nodes
    city_img = np.zeros([size_y+2, size_x+2],dtype=np.uint8)
    for node1, node2, *_ in city_graph.edges:

        p1 = pos_img[node_index_map[node1]]
        p2 = pos_img[node_index_map[node2]]
        
        rr, cc = line(p1[1], p1[0], p2[1], p2[0])
        
        city_img[rr, cc] = 1

    return city_img

def calculate_lacunarity(graphml, max_radius, delta_radius=5, px_per_km=100, return_image=False):
    """Calculate the lacunarity [1] of a city represented as a graph. Since the lacunarity
    is defined for images, the city is first drawn as a rasterized image. The function 
    returns the lacunarity calculated at the following radii:
    
    radii = np.arange(1, max_radius, delta_radius)

    [1] Rodrigues, E.P., Barbosa, M.S. and Costa, L.D.F., 2005. Self-referred approach to lacunarity. 
        Physical Review E, 72(1), p.016707.

    Parameters
    ----------
    city_graph : networkx Graph
        Graph containing attributes 'posx' (longitude) and 'posy' (latitude).
    max_radius : int
        Maximum radius (scale) to use for lacunarity.
    delta_radius : int
        Radius increment between scales.
    px_per_km : int
        Number of pixels used to represent each kilometer, that is, the resolution of the
        image generated for the city.
    return_image : bool
        Whether to also return the city image used for calculating the lacunarity

    Returns
    -------
    radii : np.ndarray
        Radius values used for lacunarity calculation
    lacunarity_values : np.ndarray
        Lacunarity values
    city_img : np.ndarray
        An image representing the streets of the city. Only returned if return_image is True
    """

    city_graph = nx.read_graphml(graphml)
    city_img = draw_city(city_graph, px_per_km)
    radii, lacunarity_values = lacunarity(city_img, max_radius, delta_radius)
    #radii = radii/px_per_km

    if return_image:
        return radii, lacunarity_values, city_img
    else:
        return radii, lacunarity_values

##########################################################
def main():
    info(inspect.stack()[0][3] + '()')
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graphml', help='Graph in graphml fmt')
    parser.add_argument('outdir', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    aux = os.path.basename(args.graphml).replace('.graphml', '.csv')
    outpath = pjoin(args.outdir, aux)

    if os.path.exists(outpath):
        info('{} already exists'.format(outpath))
        info('Aborting...')
        return

    px_per_km = 100 # To obtain per meter, divide by 1000
    max_radius = 100
    delta_radius = 5

    radii, lacunarity_values = calculate_lacunarity(city_graph, max_radius,
                                                    delta_radius, px_per_km)

    data = np.array([radii, lacunarity_values]).T
    pd.DataFrame(data, columns=['radius', 'lacunarity']).to_csv(outpath, index=False)

    info('Elapsed time:{}'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))

if __name__=='__main__':
    main()
