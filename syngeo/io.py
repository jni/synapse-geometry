# stardard library
import os
import json
import cPickle as pck
import itertools as it

# external libraries
import numpy as np
from ray import imio

def add_anything(a, b):
    return a + b

def write_synapse_to_vtk(neurons, coords, fn, im=None, margin=None):
    """Output neuron shapes around pre- and post-synapse coordinates.
    
    The coordinate array is a (n+1) x m array, where n is the number of 
    post-synaptic sites (fly neurons are polyadic) and m = neurons.ndim, the
    number of dimensions of the image.
    """
    neuron_ids = neurons[zip(*coords)]
    mean_coords = coords.mean(axis=0).astype(np.uint)
    neurons = get_box(neurons, mean_coords, margin)
    synapse_volume = reduce(add_anything, 
        [(i+1)*(neurons==j) for i, j in enumerate(neuron_ids)])
    imio.write_vtk(synapse_volume.astype(np.uint8), fn)
    if im is not None:
        im = get_box(im, mean_coords, margin)
        imio.write_vtk(im, 
            os.path.join(os.path.dirname(fn), 'image.' + os.path.basename(fn)))

def all_sites(synapses):
    return list(it.chain(*tbar_post_pairs_to_arrays(synapses)))

def all_postsynaptic_sites(synapses):
    tbars, posts = zip(*synapses)
    return list(it.chain(*posts))

def get_box(a, coords, margin):
    """Obtain a box of size 2*margin+1 around coords in array a.

    Boxes close to the boundary are trimmed accordingly.
    """
    if margin is None:
        return a
    coords = np.array(coords)[np.newaxis, :].astype(int)
    origin = np.zeros(coords.shape, dtype=int)
    shape = np.array(a.shape)[np.newaxis, :]
    topleft = np.concatenate((coords-margin, origin), axis=0).max(axis=0)
    bottomright = np.concatenate((coords+margin+1, shape), axis=0).min(axis=0)
    box = [slice(top, bottom) for top, bottom in zip(topleft, bottomright)]
    return a[box].copy()

def tbar_post_pairs_to_arrays(pairs):
    return [np.concatenate((t[np.newaxis, :], p), axis=0) for t, p in pairs]

def volume_synapse_view(pairs, shape):
    v = np.zeros(shape, int)
    for i, (pre, post) in enumerate(pairs):
        coords = np.concatenate((pre[np.newaxis, :], post), axis=0)
        coords = [coords[:, j] for j in range(coords.shape[1])]
        for j in range(len(coords)):
            if coords[j][0] < 0:
                coords[j] = shape[j] + coords[j]
        v[coords] = i+1
    return v

def synapses_from_raveler_session_data(fn, output_format='pairs', 
                                    t=(2, 1, 0), s=(1, -1, 1), transform=True):
    if not transform:
        t = (0, 1, 2)
        s = (1, 1, 1)
    with open(fn) as f:
        d = pck.load(f)
    annots = d['annotations']['point']
    tbars = [a for a in annots if annots[a]['kind'] == 'T-bar']
    posts = [annots[a] for a in tbars]
    posts = [eval(p['value'].replace('false', 'False').replace('true', 'True'))
             for p in posts]
    posts = [p['partners'] for p in posts]
    posts = [map(lambda x: x[0], p) for p in posts]
    tbars = [coord_transform(tbar, t, s) for tbar in tbars]
    posts = [coord_transform(post, t, s) for post in posts]
    pairs = zip(tbars, posts)
    if output_format == 'pairs':
        return pairs
    elif output_format == 'arrays':
        return tbar_post_pairs_to_arrays(pairs)

def raveler_synapse_annotations_to_coords(fn, output_format='pairs',
                                    t=(2, 1, 0), s=(1, -1, 1), transform=True):
    """Obtain pre- and post-synaptic coordinates from Raveler annotations."""
    if not transform:
        t = (0, 1, 2)
        s = (1, 1, 1)
    with open(fn, 'r') as f:
        syns = json.load(f)['data']
    tbars = [coord_transform(syn['T-bar']['location'], t, s) for syn in syns]
    posts = [coord_transform([p['location'] for p in syn['partners']], t, s)
            for syn in syns]
    pairs = zip(tbars, posts)
    if output_format == 'pairs':
        return pairs
    elif output_format == 'arrays':
        return tbar_post_pairs_to_arrays(pairs)

def coord_transform(coords, t=(2, 1, 0), s=(1, -1, 1)):
    coords = np.array(coords)
    s = np.array(s)
    return coords[..., t]*s - (s == -1)

def write_all_synapses_to_vtk(neurons, list_of_coords, fn, im, margin=None,
                                                            single_pairs=True):
    for i, coords in enumerate(list_of_coords):
        if single_pairs:
            pre = coords[0]
            for j, post in enumerate(coords[1:]):
                pair_coords = np.concatenate(
                    (pre[np.newaxis, :], post[np.newaxis, :]), axis=0)
                cfn = fn%(i, j)
                write_synapse_to_vtk(neurons, pair_coords, cfn, im, margin)
        else:
            cfn = fn%i
            write_synapse_to_vtk(neurons, coords, cfn, im, margin)
