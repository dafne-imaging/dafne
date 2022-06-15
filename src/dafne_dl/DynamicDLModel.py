#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of a deep learning module that can be serialized and deserialized, and dynamically changed.
Functions for the operation of the class are provided as references to top-level functions.
Such top level functions should define all the imports within themselves (i.e. don't put the imports at the top of the file).
"""
#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from .interfaces import DeepLearningClass
import dill
from io import BytesIO
import numpy as np
import inspect


def fn_to_source(function):
    """
    Given a function, returns it source. If the source cannot be retrieved, return the object itself
    """
    #print('Converting fn to source')
    if function is None: return None
    try:
        return inspect.getsource(function)
    except OSError:
        #print('Conversion failed!')
        try:
            src = function.source
            #print('Returning embedded source')
            return src
        except:
            pass
    print('Getting source failed - Returning bytecode')
    return function # the source cannot be retrieved, return the object itself


def source_to_fn(source):
    """
    Given a source, return the (first) defined function. If the source is not a string, return the object itself
    """
    if type(source) is not str:
        print("source to fn: source is not a string")
        return source
    #print("source to fn: source is string")
    locs = {}
    globs = {}
    try:
        exec(source, globs, locs)
    except:
        return source # the string was just a string apparently, not valid code
    for k,v in locs.items():
        if callable(v):
            #print('source_to_fn. Found function', k)
            v.source = source
            return v
    return source


def default_keras_weights_to_model_function(modelObj: DynamicDLModel, weights):
    modelObj.model.set_weights(weights)


def default_keras_model_to_weights_function(modelObj: DynamicDLModel):
    return modelObj.model.get_weights()


def default_keras_delta_function(lhs: DynamicDLModel, rhs: DynamicDLModel, threshold=None):
    from ..dl.interfaces import IncompatibleModelError
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        delta = lhs_weights[depth] - rhs_weights[depth]
        if threshold is not None:
            delta[np.abs(delta) < threshold] = 0
        newWeights.append(delta)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    outputObj.is_delta = True
    outputObj.timestamp_id = rhs.timestamp_id # set the timestamp of the original model to identify the base
    return outputObj


def default_keras_add_weights_function(lhs: DynamicDLModel, rhs: DynamicDLModel):
    from ..dl.interfaces import IncompatibleModelError
    if lhs.model_id != rhs.model_id: raise IncompatibleModelError
    lhs_weights = lhs.get_weights()
    rhs_weights = rhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] + rhs_weights[depth])
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj


def default_keras_multiply_function(lhs: DynamicDLModel, rhs: float):
    if not isinstance(rhs, (int, float)):
        raise NotImplementedError('Incompatible types for multiplication (only multiplication by numeric factor is allowed)')
    lhs_weights = lhs.get_weights()
    newWeights = []
    for depth in range(len(lhs_weights)):
        newWeights.append(lhs_weights[depth] * rhs)
    outputObj = lhs.get_empty_copy()
    outputObj.set_weights(newWeights)
    return outputObj


def default_keras_weight_copy_function(weights_in):
    weights_out = []
    for layer in weights_in:
        weights_out.append(layer.copy())
    return weights_out


class DynamicDLModel(DeepLearningClass):

    """
    Class to represent a deep learning model that can be serialized/deserialized
    """
    def __init__(self, model_id,  # a unique ID to avoid mixing different models
                 init_model_function,  # inits the model. Accepts no parameters and returns the model
                 apply_model_function,  # function that applies the model. Has the object, and image, and a sequence containing resolutions as parameters
                 weights_to_model_function = default_keras_weights_to_model_function,  # put model weights inside the model.
                 model_to_weights_function = default_keras_model_to_weights_function,  # get the weights from the model in a pickable format
                 calc_delta_function = default_keras_delta_function,  # calculate the weight delta
                 apply_delta_function = default_keras_add_weights_function,  # apply a weight delta
                 weight_copy_function = default_keras_weight_copy_function,  # create a deep copy of weights
                 factor_multiply_function = default_keras_multiply_function,
                 incremental_learn_function = None,  # function to perform an incremental learning step
                 weights = None,  # initial weights
                 timestamp_id = None,
                 is_delta = False):
        self.model = None
        self.model_id = model_id
        self.is_delta = is_delta

        # lsit identifying the external functions that need to be saved with source and serialized
        self.function_mappings = [
            'init_model_function',
            'apply_model_function',
            'weights_to_model_function',
            'model_to_weights_function',
            'calc_delta_function',
            'apply_delta_function',
            'weight_copy_function',
            'factor_multiply_function',
            'incremental_learn_function',
        ]

        # the following sets the internal attributes self.fn = fn, with additionally adding the source to the function
        for fn_name in self.function_mappings:
            self.set_internal_fn(fn_name, locals()[fn_name])


        self.init_model() # initializes the model
        if timestamp_id is None:
            self.reset_timestamp()
        else:
            self.timestamp_id = timestamp_id  # unique timestamp id; used to identify model versions during federated learning

        if weights: self.set_weights(weights)

    def set_internal_fn(self, internal_name, obj):
        #print('Setting', internal_name)
        if callable(obj):
            src = fn_to_source(obj)
            if type(src) == str:
                obj.source = src

        setattr(self, internal_name, obj)

    def init_model(self):
        """
        Initializes the internal model

        Returns
        -------
        None.

        """
        self.model = self.init_model_function()
        
    def set_weights(self, weights):
        """
        Loads the weights in the internal model

        Parameters
        ----------
        weights : whatever is accepted by the model_to_weights_function
            Weights to be loaded into the model

        Returns
        -------
        None.

        """
        self.weights_to_model_function(self, weights)
        
    def get_weights(self):
        return self.model_to_weights_function(self)
        
    def apply_delta(self, other):
        return self.apply_delta_function(self, other)
    
    def calc_delta(self, other, threshold=None):
        return self.calc_delta_function(self, other, threshold)
    
    def apply(self, data):
        return self.apply_model_function(self, data)

    def factor_multiply(self, factor: float):
        return self.factor_multiply_function(self, factor)
    
    def incremental_learn(self, trainingData, trainingOutputs, bs=5, minTrainImages=5):
        self.incremental_learn_function(self, trainingData, trainingOutputs, bs, minTrainImages)
        
    def dump(self, file):
        """
        Dumps the current status of the object, including functions and weights
        
        Parameters
        ----------
        file:
            a file descriptor (open in writable mode)

        Returns
        -------
        Nothing

        """
        outputDict = {
            'model_id': self.model_id,
            'weights': self.get_weights(),
            'timestamp_id': self.timestamp_id,
            'is_delta': self.is_delta
            }

        # add the internal functions to the dictionary
        for fn_name in self.function_mappings:
            outputDict[fn_name] = fn_to_source(getattr(self, fn_name))

        dill.dump(outputDict, file)
    
    def dumps(self) -> bytes:
        file = BytesIO()
        self.dump(file)
        return file.getvalue()
    
    def get_empty_copy(self) -> DynamicDLModel:
        """
        Gets an empty copy (i.e. without weights) of the current object

        Returns
        -------
        DynamicDLModel
            Output copy

        """
        new_model = DynamicDLModel(self.model_id, self.init_model_function, self.apply_model_function,
                                   weights=None, timestamp_id=self.timestamp_id, is_delta=self.is_delta)
        for fn_name in self.function_mappings:
            new_model.set_internal_fn(fn_name, getattr(self, fn_name))
        return new_model

    def copy(self) -> DynamicDLModel:
        """
        Gets a copy (i.e. with weights) of the current object

        Returns
        -------
        DynamicDLModel
            Output copy

        """
        model_out = self.get_empty_copy()
        model_out.set_weights( self.weight_copy_function(self.get_weights()) )
        return model_out

    @staticmethod
    def Load(file) -> DynamicDLModel:
        """
        Creates an object from a file

        Parameters
        ----------
        file : file descriptor
            A file descriptor.

        Returns
        -------
        DynamicDLModel
            A new instance of a dynamic model

        """
        
        inputDict = dill.load(file)
        for k,v in inputDict.items():
            if '_function' in k:
                inputDict[k] = source_to_fn(v) # convert the functions from source

        #print(inputDict)
        outputObj = DynamicDLModel(**inputDict)
        return outputObj
        
    @staticmethod
    def Loads(b: bytes) -> DynamicDLModel:
        """
        Creates an object from a binary dump

        Parameters
        ----------
        file : bytes
            A sequence of bytes

        Returns
        -------
        DynamicDLModel
            A new instance of a dynamic model

        """
        file = BytesIO(b)
        return DynamicDLModel.Load(file)
