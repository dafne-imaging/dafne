#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from abc import ABC, abstractmethod
from io import BytesIO
from typing import IO, Callable, List, Union, Optional, Dict
import numpy as np
import time


class IncompatibleModelError(Exception):
    pass

class DeepLearningClass(ABC):   
    
    @abstractmethod
    def init_model(self):
        """
        Initializes the model when needed

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod
    def calc_delta(self, baseModel: DeepLearningClass) -> DeepLearningClass:
        """
        Calculate a delta with another model, Returns a new instance

        Parameters
        ----------
        baseModel : DeepLearningClass
            Base model to calculate the delta from

        Returns
        -------
        DeepLearningClass
            A deep learning class representing the delta of the two models

        """
        pass
    
    def __sub__(self, rhs):
        return self.calc_delta(rhs)
    
    @abstractmethod
    def apply_delta(self, delta_model: DeepLearningClass) -> DeepLearningClass:
        """
        Applies a delta to this class and returns a new model with the delta applied
        

        Parameters
        ----------
        delta_model : DeepLearningClass
            Applies a delta to the current model

        Returns
        -------
        DeepLearningClass
            The model that is the current model plus the delta

        """
        pass

    def sum(self, other):
        """
        Note: this defaults to apply_delta. Redefine to change behavior
        """
        return self.apply_delta(other)

    def __add__(self, rhs):
        return self.sum(rhs)
    
    @abstractmethod
    def incremental_learn(self, training_data: dict, training_outputs: str):
        """
        Perform an incremental learning step on the given training data/outputs

        Parameters
        ----------
        training_data : Dictionary
            Contains the path to the training data and resolution.
        training_outputs : String
            Contains the path to the training labels.

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod
    def apply(self, data: dict):
        """
        Applies the deep learning model to the image

        Parameters
        ----------
        data : Dictionary
            Contains the data and the extra information (for example image and resolution)

        Returns
        -------
        Depends on the operation performed:
            For classifiers: str - Containing the label of the image
            For segmenters: dict[str, mask] - Containing the labels and the corresponding 2D masks

        """
        pass

    @abstractmethod
    def factor_multiply(self, factor: float):
        """
        Multiplies all the weights by a float factor
        
        """
        pass

    def __mul__(self, factor: float):
        if not isinstance(factor, (int, float)):
            raise NotImplementedError('Incompatible types for multiplication (only multiplication by numeric factor is allowed)')
        return self.factor_multiply(factor)

    def __rmul__(self, factor: float):
        if not isinstance(factor, (int, float)):
            raise NotImplementedError('Incompatible types for multiplication (only multiplication by numeric factor is allowed)')
        return self.factor_multiply(factor)

    def __call__(self, data: dict):
        return self.apply(data)

    def reset_timestamp(self):
        self.timestamp_id = int(time.time())

class ModelProvider(ABC):
    """
        Abstract class that is the base for loading (and, in the future, storing?) models.
        Has to be subclassed to support local and remote loading.
    """
    
    @abstractmethod
    def load_model(self, model_name: str, progress_callback: Optional[Callable[[int, int], None]] = None,
                   force_download: bool = False,
                   timestamp: Optional[Union[int,str]] = None) -> DeepLearningClass:
        """
        Loads a deep learning model.

        Parameters
        ----------
        model_name : str
            The name of the model to load.
        progress_callback: Callable[[int, int], None] (optional)
            Callback function for progress
        force_download: bool
            Sets the forced redownload of models
        timestamp: int or None
            Return a specific model version (default: latest)

        Returns
        -------
        The model object.

        """
        pass

    @abstractmethod
    def model_details(self, model_name: str) -> dict:
        pass

    @abstractmethod
    def available_models(self) -> Optional[list[str]]:
        """
        Parameters
        ----------
        None

        Returns
        -------
        List of available models
        """
        pass

    @abstractmethod
    def upload_model(self, model_name: str, model: DeepLearningClass, dice_score: float=0.0) -> None:
        """
        Parameters
        ----------
        model_name : str
            The name of the model to upload.
        model:
            The model to be uploaded
        dice_score:
            The average dice score of the client
        """
        pass

    def upload_data(self, data: dict) -> None:
        """
        Uploads data to the server. Converts the data into a stream before calling _upload_bytes

        Parameters
        ----------
        data: dict
            Dictionary of objects that can be saved by Numpy and loaded without using pickle (which is unsafe)
        """
        bytes_io = BytesIO()
        np.savez_compressed(bytes_io, **data)
        self._upload_bytes(bytes_io)
        bytes_io.close()


    @abstractmethod
    def _upload_bytes(self, data: IO):
        """
        Uploads generic data to the server. This is an internal function that implements the server communication.
        The actual function to be called by the client is upload_data with a dict

        Parameters
        ----------
        data: IO
            byte stream that is sent to the server.
        """
        pass

    @abstractmethod
    def log(self, msg: str):
        """
        Sends a message to the server to be logged

        Parameters
        ----------
        msg: str
            the message.
        """
        pass