# Import libraries
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, bundle_graph
import datashader as ds
import datashader.transfer_functions as tf
from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
import colorcet as cc

hv.extension('bokeh')
defaults = dict(width = 800, height = 800)

kwargs = dict(width = 800, height = 800, xaxis = None, yaxis = None)
opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))

colors = ['#000000'] + hv.Cycle('Category20').values

hv.opts.defaults(opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))

cvsopts = dict(plot_height = 600, plot_width = 600)

# Functions for plotting w/ datashader
def nodesplot(nodes, name = None, canvas = None, cat = None):
    
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    aggregator = None if cat is None else ds.count_cat(cat)
    agg = canvas.points(nodes,'x','y',aggregator)
    
    return tf.spread(tf.shade(agg, cmap = ["#FF3333"]), px = 3, name = name)

def edgesplot(edges, name = None, canvas = None):
    
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    return tf.shade(canvas.line(edges, 'x','y', agg = ds.count()), name = name)
    
def graphplot(nodes, edges, name = "", canvas = None, cat = None):
    
    if canvas is None:
        
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        
        canvas = ds.Canvas(x_range = xr, y_range = yr, **cvsopts)
        
    np = nodesplot(nodes, name + " nodes", canvas, cat)
    ep = edgesplot(edges, name + " edges", canvas)
    
    return tf.stack(ep, np, how = "over", name = name)

# Function to remove edges
def remove_edges(graph):
    
    vals = nx.edge_betweenness_centrality(graph)
    edge = ()

    # Extract the edge with highest edge betweenness centrality score
    for key, value in sorted(vals.items(), key = lambda item: item[1], reverse = True):
      
        edge = key
      
        break

    return edge

# Function to build communtiies
def build_communities(graph):

    sg = nx.connected_components(graph)
    sg_count = nx.number_connected_components(graph)

    while(sg_count == 1):
        
        graph.remove_edges(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
        sg = nx.connected_components(graph)
        sg_count = nx.number_connected_components(graph)

    return sg

def draw_graph(graph, pos, lista, listb, measure_name):
    
    nodes = nx.draw_networkx_nodes(graph, pos, node_size = 100, cmap = cc.bmw, node_color = lista, nodelist = listb)
    nodes.set_norm(mcolors.SymLogNorm(linthresh = 0.01, linscale = 1))
    
    edges = nx.draw_networkx_edges(G, pos)
    
    plt.title(measure_name, fontsize = 22, fontname = 'Arial')
    plt.colorbar(nodes)
    plt.axis('off')