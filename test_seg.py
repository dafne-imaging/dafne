#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:56:36 2020

@author: francesco
"""
from test.testDL import testSegmentation
from test.testDL import testClassification

#testSegmentation('models/thigh.model', 'testImages/thigh_test.dcm')
#testSegmentation('models/leg.model', 'testImages/leg_test.dcm')

testClassification('models/classifier.model', 'testImages/thigh_test.dcm')
testClassification('models/classifier.model', 'testImages/leg_test.dcm')