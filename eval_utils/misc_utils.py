import os, sys, pickle
import numpy as np
from matplotlib import pyplot as plt


def plot_figures(figures, nrows=5, ncols=5):
    '''
        Plot figures in 5x5 format
        input: torch tensor
    '''
    figures = figures[:25]
    figures = np.moveaxis(np.hstack((figures.detach().cpu().numpy()+1)/2),0,-1)
    plt.imshow(np.hstack(np.array(np.split(figures, 5))))

def save_obj(obj, name):
    '''
        Pickle objects
        input: object, filename
    '''
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    '''
        un-pickle objects
        input: filename
        returns: object
    '''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
