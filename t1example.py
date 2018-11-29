#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:16:56 2018

@author: Francesco Santini <francesco.santini@unibas.ch>

Released under a MIT license. See LICENSE for details.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from utils import pickPoint, showFit, fitError

############################
###         T1           ###
############################

# This dateset is an inversion-recovery turbo spin echo (real-part data), TR=15000ms
t1dataset = np.load('t1dataset.npy')
ti = np.array([50., 100., 200., 500., 2000., 4000.])

# create magnitude data (this is what you will often have)
t1dataset = np.abs(t1dataset)

# the negative part of the curve is now positive

# pick one point from an image and show the time evolution
timeSeries = pickPoint(t1dataset, ti)

# this is the classical T1 model function
# you can play around with this. Maybe you can add a non-oerfect inversion term
# i.e. change the 2 to a free parameter
def t1ModelFunction(ti, m0, t1):
    return m0*(1 - 2*np.exp(-ti/t1))

# The first part of the curve must be negative. Find the minimum value
minPoint = np.argmin(timeSeries)

timeSeries1 = np.copy(timeSeries)
# everything up to the minimum must be negative. The only question is the minimum
# first, fit the curve with the minimum point as positive
timeSeries1[0:minPoint] = -timeSeries1[0:minPoint]

fittedParams1, covariance = opt.curve_fit(t1ModelFunction, ti, timeSeries1, p0 = (100,1000))

# m0 is the "proton density", actually containing everything that is not T1,
# i.e. coil sensitivity, etc
m0 = fittedParams1[0]
t1 = fittedParams1[1]

print("T1: {}, Param error {}".format(t1, fitError(covariance)))

# show the fit
showFit(ti, timeSeries1, t1ModelFunction, fittedParams1)

# repeat the fit by assuming the minimum point as negative
timeSeries2 = np.copy(timeSeries)
timeSeries2[0:minPoint+1] = -timeSeries2[0:minPoint+1]

fittedParams2, covariance = opt.curve_fit(t1ModelFunction, ti, timeSeries2, p0 = (100,1000))

# m0 is the "proton density", actually containing everything that is not T1,
# i.e. coil sensitivity, etc
m0 = fittedParams2[0]
t1 = fittedParams2[1]

print("T1: {}, Param error {}".format(t1, fitError(covariance)))

# show the fit
showFit(ti, timeSeries2, t1ModelFunction, fittedParams2)

# How would you choose the most "correct" T1?

# Exercise: based on the simple example, calculate a T1 map over the whole image
# Hint: iterate over the first two dimensions. To improve speed, try setting
# a noise threshold.

