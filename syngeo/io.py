# stardard library
import sys, os
import json
import cPickle as pck
import itertools as it

# external libraries
import numpy as np
from scipy.ndimage import distance_transform_edt as distance_transform
from scipy.spatial.distance import euclidean as euclidean_distance
from ray import imio, evaluate, morpho

def add_anything(a, b):
    return a + b

def write_synapse_to_vtk(a, coords, fn, im=None, margin=None):
    """Output neuron shapes around pre- and post-synapse coordinates.
    
    The coordinate array is a (n+1) x m array, where n is the number of 
    post-synaptic sites (fly neurons are polyadic) and m = a.ndim, the
    number of dimensions of the image.
    """
    neuron_ids = a[zip(*coords)]
    mean_coords = coords.mean(axis=0).astype(np.uint)
    write_neurons_to_vtk(a, neuron_ids, mean_coords, fn, im, margin)

def write_neurons_to_vtk(a, ids, center, fn, im=None, margin=None):
    """Output neuron shapes around a center-point, given neuron ids."""
    a = get_box(a, center, margin)
    synapse_volume = reduce(add_anything, 
        [(i+1)*(a==j) for i, j in enumerate(neuron_ids)])
    imio.write_vtk(synapse_volume.astype(np.uint8), fn)
    if im is not None:
        im = get_box(im, mean_coords, margin)
        imio.write_vtk(im, 
            os.path.join(os.path.dirname(fn), 'image.' + os.path.basename(fn)))

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

def write_all_candidates_to_vtk(neurons, pairs, fn, im, threshold=350,
                                                    scale=10, margin=100):
    _, false_candidates, true_ids, _ = \
        candidate_postsynaptics(neurons, pairs, threshold, scale)
    for i, (pre, posts), fcs, trues in \
                    enumerate(zip(pairs, false_candidates, true_ids):
        truth = [True]*len(trues) + [False]*len(fcs)
        psds = trues + fcs
        pre_id = neurons[tuple(pre)]
        for j, (psd, t) in enumerate(zip(psds, truth)):
            cfn = fn % (i, j, int(t))
            write_neurons_to_vtk(neurons, [pre_id, psd], pre, cfn, im, margin)

def all_sites(synapses):
    return list(it.chain(*tbar_post_pairs_to_arrays(synapses)))

def all_postsynaptic_sites(synapses):
    tbars, posts = zip(*synapses)
    return list(it.chain(*posts))

def candidate_postsynaptics(a, synapses, threshold=350, scale=10):
    candidates = []
    false_candidates = []
    true_ids = []
    true_candidates = []
    v = volume_presynapse_view(synapses, a.shape)
    for i, (pre, posts) in enumerate(synapses):
        pre_id = a[tuple(pre)]
        post_ids = a[tuple(posts.T)]
        local_ids = get_box(a, pre, threshold/scale)
        local_syn = get_box(v, pre, threshold/scale)
        pix = scale * distance_transform(local_syn != i+1) < threshold
        cand = np.setdiff1d(np.unique(local_ids[pix]), np.array([pre_id]))
        candidates.append(cand)
        false_candidates.append(np.setdiff1d(cand, post_ids))
        true_ids.append(post_ids)
        true_candidates.append(np.setdiff1d(post_ids, cand))
    return candidates, false_candidates, true_ids, true_candidates

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

def volume_presynapse_view(pairs, shape):
    v = np.zeros(shape, int)
    for i, (pre, post) in enumerate(pairs):
        v[tuple(pre)] = i+1
    return v

def volume_synapse_view(pairs, shape):
    v = np.zeros(shape, int)
    for i, (pre, post) in enumerate(pairs):
        coords = np.concatenate((pre[np.newaxis, :], post), axis=0)
        coords = [coords[:, j] for j in range(coords.shape[1])]
        v[coords] = i+1
    return v

def synapses_to_annotations_json(syn, fn,
                                    t=(2, 1, 0), s=(1, -1, 1), transform=True):
    data = []
    for pre, posts in syn:
        synapse = {
            'T-bar': {
                'location': map(int, pre[list(t)] * s),
                'body ID': -1,
                'status': 'final',
                'confidence': 1.0
            },
            'partners': [
                {'location': map(int, post[list(t)] * s),
                'body ID': -1, 'confidence': 1.0} for post in posts
            ]
        }
        data.append(synapse)
    metadata = {'description': 'synapse annotations'}
    with open(fn, 'w') as f:
        json.dump({'data':data, 'metadata':metadata}, f, indent=4)

def annotations_json_to_synapses(fn, output_format='pairs',
                                    t=(2, 1, 0), s=(1, -1, 1), transform=True):
    """Obtain pre- and post-synaptic coordinates from Raveler annotations."""
    if not transform:
        t = (0, 1, 2)
        s = (1, 1, 1)
    with open(fn, 'r') as f:
        syns = json.load(f)['data']
    tbars = [np.array(syn['T-bar']['location'])[:, t]*s for syn in syns]
    posts = [np.array([p['location'] for p in syn['partners']])[:, t]*s 
        for syn in syns]
    pairs = zip(tbars, posts)
    if output_format == 'pairs':
        return pairs
    elif output_format == 'arrays':
        return tbar_post_pairs_to_arrays(pairs)

def session_data_to_synapses(fn, output_format='pairs', 
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
    tbars = [np.array(tbar)[np.array(t)]*s for tbar in tbars]
    posts = [np.array(post)[:, t] * s for post in posts]
    pairs = zip(tbars, posts)
    if output_format == 'pairs':
        return pairs
    elif output_format == 'arrays':
        return tbar_post_pairs_to_arrays(pairs)

def raveler_to_numpy_coord_transform(coords, t=(2, 1, 0), s=(1, -1, 1)):
    coords = np.array(coords)
    return coords[:, t]*s

