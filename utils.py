#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:38:33 2018

@author: Francesco Santini <francesco.santini@unibas.ch>

Released under a MIT license. See LICENSE for details.
"""

import numpy as np
import matplotlib.pyplot as plt

def pickPoint(dataset, timebase):
    plt.imshow(dataset[:,:,0])
    plt.title("Click on a point to show the time evolution")
    p = plt.ginput()
    timeseries = np.squeeze(dataset[int(p[0][1]), int(p[0][0]), :])
    plt.clf() # clear figure
    plt.plot(timebase, timeseries)
    return timeseries

def showFit(timebase, yValues, modelFun, params):
    plt.close()
    plt.plot(timebase, yValues, 'bo')
    t = np.linspace(timebase[0], timebase[-1])
    fitted = np.array(list(map(lambda x: modelFun(x, *params), t)))
    plt.plot(t, fitted, 'k')
    plt.title("Close the figure to continue")
    plt.show()
    
def fitError(pcov):
    return np.sqrt(np.diag(pcov))
