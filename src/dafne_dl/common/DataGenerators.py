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

# Data generator classes - from directory or from memory
#from collections import Sequence
from tensorflow.keras.utils import Sequence
import numpy as np
import os

# This is NOT a general implementation! See the number of masks below
class DataGeneratorDir(Sequence):
    def __init__(self, path, list_X=list(range(1, 4501)), batch_size=20, dim=(432, 432), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_X = list_X
        self.path = path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_X_temp = [self.list_X[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_X_temp, self.path)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_X_temp, path):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 2))
        y = np.empty((self.batch_size, *self.dim, 13)) # chenge this to be generic!

        # Generate data
        for i, j in enumerate(list_X_temp):
            # Store sample
            arr = np.load(os.path.join(path, 'train_' + str(j) + '.npy'))

            X[i,] = np.stack([arr[:, :, 0], arr[:, :, -1]], axis=-1)

            # Store class
            y[i,] = arr[:, :, 1:-1]

        return X, y

class DataGeneratorMem(Sequence):
    def __init__(self, training_data_list, list_X=list(range(1, 4501)), batch_size=20, dim=(432, 432), shuffle=True):
        print('Data Generator Initialization. Data list len:', len(training_data_list))
        print('list_X', list_X)
        self.dim = dim
        self.batch_size = batch_size
        self.list_X = list_X
        self.training_data_list = training_data_list
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n_labels = training_data_list[0].shape[2]-2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_X_temp = [self.list_X[k] for k in indexes]
        print("list_X_temp", list_X_temp)

        # Generate data
        X, y = self.__data_generation(list_X_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_X_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 2))
        y = np.empty((self.batch_size, *self.dim, self.n_labels))

        # Generate data
        for i, j in enumerate(list_X_temp):
            # Store sample
            arr = self.training_data_list[j]

            #X[i,] = np.stack([arr[:, :, 0], arr[:, :, -1]], axis=-1)
            X[i, :, :, 0] = arr[:, :, 0]
            X[i, :, :, 1] = arr[:, :, -1]

            # Store class
            y[i,] = arr[:, :, 1:-1]

        return X, y
