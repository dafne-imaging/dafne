#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from test.testDL import testSegmentation
from test.testDL import testClassification

#testSegmentation('models/thigh.model', 'testImages/thigh_test.dcm')
#testSegmentation('models/leg.model', 'testImages/leg_test.dcm')

testClassification('models/classifier.model', 'testImages/thigh_test.dcm')
testClassification('models/classifier.model', 'testImages/leg_test.dcm')