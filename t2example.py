#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:16:10 2018

@author: Francesco Santini <francesco.santini@unibas.ch>

Released under a MIT license. See LICENSE for details.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from utils import pickPoint, showFit, fitError

############################
###         T2           ###
############################

# This dataset is a single-echo spin echo, TR=5000ms
t2dataset = np.load('t2dataset.npy')
te = np.array([10., 25., 50., 90., 150., 250.])

# pick one point from an image and show the time evolution
timeSeries = pickPoint(t2dataset, te)

# this is the classical T2 model function
# you can play around with this. Maybe you can add a noise term
def t2ModelFunction(te, m0, t2):
    return m0*np.exp(-te/t2)

fittedParams, covariance = opt.curve_fit(t2ModelFunction, te, timeSeries, p0 = (100,100))

# m0 is the "proton density", actually containing everything that is not T1,
# i.e. coil sensitivity, etc
m0 = fittedParams[0]
t2 = fittedParams[1]

print("T2: {}, Param error {}".format(t2, fitError(covariance)))

# show the fit
showFit(te, timeSeries, t2ModelFunction, fittedParams)

# Exercise: based on the simple example, calculate a T1 map over the whole image
# Hint: iterate over the first two dimensions. To improve speed, try setting
# a noise threshold.
