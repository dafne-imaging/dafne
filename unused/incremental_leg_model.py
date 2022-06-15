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

from src.dafne_dl import DynamicDLModel
import numpy as np # this is assumed to be available in every context

def gamba_unet():
    
    from tensorflow.keras import regularizers
    from tensorflow.keras.activations import softmax
    from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Concatenate, Lambda, Activation, Reshape, Add
    from tensorflow.keras.models import Model

    inputs=Input(shape=(432,432,2))
    weight_matrix=Lambda(lambda z: z[:,:,:,1])(inputs)
    weight_matrix=Reshape((432,432,1))(weight_matrix)
    reshape=Lambda(lambda z : z[:,:,:,0])(inputs)
    reshape=Reshape((432,432,1))(reshape)

    reg=0.01
    
    #reshape=Dropout(0.2)(reshape)   ## Hyperparameter optimization only on visible layer
    Level1_l=Conv2D(filters=32,kernel_size=(1,1),strides=1,kernel_regularizer=regularizers.l2(reg))(reshape)
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    Level1_l_shortcut=Level1_l#Level1_l#
    Level1_l=Activation('relu')(Level1_l)
    Level1_l=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_l)#(Level1_l)# ##  kernel_initializer='glorot_uniform' is the default
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    #Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_l=Activation('relu')(Level1_l)
    #Level1_l=Dropout(0.5)(Level1_l)   
    Level1_l=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_l)
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    #Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_l=Add()([Level1_l,Level1_l_shortcut])
    Level1_l=Activation('relu')(Level1_l)


    Level2_l=Conv2D(filters=64,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level1_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    Level2_l_shortcut=Level2_l
    Level2_l=Activation('relu')(Level2_l)
    Level2_l=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=InstanceNormalization(axis=-1)(Level2_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_l=Activation('relu')(Level2_l)
    #Level2_l=Dropout(0.5)(Level2_l)
    Level2_l=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=InstanceNormalization(axis=-1)(Level2_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_l=Add()([Level2_l,Level2_l_shortcut])
    Level2_l=Activation('relu')(Level2_l)
    
    
    Level3_l=Conv2D(filters=128,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    Level3_l_shortcut=Level3_l
    Level3_l=Activation('relu')(Level3_l)
    Level3_l=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    #Level3_l=InstanceNormalization(axis=-1)(Level3_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_l=Activation('relu')(Level3_l)
    #Level3_l=Dropout(0.5)(Level3_l)
    Level3_l=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    #Level3_l=InstanceNormalization(axis=-1)(Level3_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_l=Add()([Level3_l,Level3_l_shortcut])
    Level3_l=Activation('relu')(Level3_l)
    
    
    Level4_l=Conv2D(filters=256,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    Level4_l_shortcut=Level4_l
    Level4_l=Activation('relu')(Level4_l)
    Level4_l=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    #Level4_l=InstanceNormalization(axis=-1)(Level4_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_l=Activation('relu')(Level4_l)
    #Level4_l=Dropout(0.5)(Level4_l)
    Level4_l=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    #Level4_l=InstanceNormalization(axis=-1)(Level4_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_l=Add()([Level4_l,Level4_l_shortcut])
    Level4_l=Activation('relu')(Level4_l)


    Level5_l=Conv2D(filters=512,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    Level5_l_shortcut=Level5_l
    Level5_l=Activation('relu')(Level5_l)
    Level5_l=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_l=Activation('relu')(Level5_l)
    #Level5_l=Dropout(0.5)(Level5_l)
    Level5_l=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_l=Add()([Level5_l,Level5_l_shortcut])
    Level5_l=Activation('relu')(Level5_l)


    Level6_l=Conv2D(filters=1024,kernel_size=(3,3),strides=3,kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level6_l=BatchNormalization(axis=-1)(Level6_l)
    Level6_l_shortcut=Level6_l
    Level6_l=Activation('relu')(Level6_l)
    Level6_l=Conv2D(filters=1024,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level6_l)
    Level6_l=BatchNormalization(axis=-1)(Level6_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level6_l=Activation('relu')(Level6_l)
    #Level5_l=Dropout(0.5)(Level5_l)
    Level6_l=Conv2D(filters=1024,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level6_l)
    Level6_l=BatchNormalization(axis=-1)(Level6_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level6_l=Add()([Level6_l,Level6_l_shortcut])
    Level6_l=Activation('relu')(Level6_l)
    
    Level5_r=Conv2DTranspose(filters=512,kernel_size=(3,3),strides=3,kernel_regularizer=regularizers.l2(reg))(Level6_l)
    Level5_r=BatchNormalization(axis=-1)(Level5_r)
    Level5_r_shortcut=Level5_r
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_r=Activation('relu')(Level5_r)
    merge5=Concatenate(axis=-1)([Level5_l,Level5_r])
    Level5_r=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge5)
    Level5_r=BatchNormalization(axis=-1)(Level5_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_r=Activation('relu')(Level5_r)
    #Level4_r=Dropout(0.5)(Level4_r)
    Level5_r=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_r)
    Level5_r=BatchNormalization(axis=-1)(Level5_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_r=Add()([Level5_r,Level5_r_shortcut])
    Level5_r=Activation('relu')(Level5_r)

    
    Level4_r=Conv2DTranspose(filters=256,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level5_r)
    Level4_r=BatchNormalization(axis=-1)(Level4_r)
    Level4_r_shortcut=Level4_r
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_r=Activation('relu')(Level4_r)
    merge4=Concatenate(axis=-1)([Level4_l,Level4_r])
    Level4_r=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge4)
    Level4_r=BatchNormalization(axis=-1)(Level4_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_r=Activation('relu')(Level4_r)
    #Level4_r=Dropout(0.5)(Level4_r)
    Level4_r=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_r)
    Level4_r=BatchNormalization(axis=-1)(Level4_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_r=Add()([Level4_r,Level4_r_shortcut])
    Level4_r=Activation('relu')(Level4_r)
    
    
    Level3_r=Conv2DTranspose(filters=128,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level4_r)
    Level3_r=BatchNormalization(axis=-1)(Level3_r)
    Level3_r_shortcut=Level3_r
    #Level3_r=InstanceNormalization(axis=-1)(Level3_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_r=Activation('relu')(Level3_r)
    merge3=Concatenate(axis=-1)([Level3_l,Level3_r])
    Level3_r=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge3)
    Level3_r=BatchNormalization(axis=-1)(Level3_r)
    #Level3_r=InstanceNormalization(axis=-1)(Level3_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_r=Activation('relu')(Level3_r)
    #Level3_r=Dropout(0.5)(Level3_r)
    Level3_r=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_r)
    Level3_r=BatchNormalization(axis=-1)(Level3_r)
    #Level3_r=InstanceNormalization(axis=-1)(Level3_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_r=Add()([Level3_r,Level3_r_shortcut])
    Level3_r=Activation('relu')(Level3_r)
    
    
    Level2_r=Conv2DTranspose(filters=64,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level3_r)
    Level2_r=BatchNormalization(axis=-1)(Level2_r)
    Level2_r_shortcut=Level2_r
    #Level2_r=InstanceNormalization(axis=-1)(Level2_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_r=Activation('relu')(Level2_r)
    merge2=Concatenate(axis=-1)([Level2_l,Level2_r])
    Level2_r=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge2)
    Level2_r=BatchNormalization(axis=-1)(Level2_r)
    #Level2_r=InstanceNormalization(axis=-1)(Level2_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_r=Activation('relu')(Level2_r)
    #Level2_r=Dropout(0.5)(Level2_r)
    Level2_r=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_r)
    Level2_r=BatchNormalization(axis=-1)(Level2_r)
    #Level2_r=InstanceNormalization(axis=-1)(Level2_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_r=Add()([Level2_r,Level2_r_shortcut])
    Level2_r=Activation('relu')(Level2_r)
    
    
    Level1_r=Conv2DTranspose(filters=32,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level2_r)
    Level1_r=BatchNormalization(axis=-1)(Level1_r)
    Level1_r_shortcut=Level1_r
    #Level1_r=InstanceNormalization(axis=-1)(Level1_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_r=Activation('relu')(Level1_r)
    merge1=Concatenate(axis=-1)([Level1_l,Level1_r])
    Level1_r=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge1)
    Level1_r=BatchNormalization(axis=-1)(Level1_r)
    #Level1_r=InstanceNormalization(axis=-1)(Level1_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_r=Activation('relu')(Level1_r)
    #Level1_r=Dropout(0.5)(Level1_r)
    Level1_r=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_r)
    Level1_r=BatchNormalization()(Level1_r)
    #Level1_r=InstanceNormalization(axis=-1)(Level1_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_r=Add()([Level1_r,Level1_r_shortcut])
    Level1_r=Activation('relu')(Level1_r)
    output=Conv2D(filters=7,kernel_size=(1,1),strides=1,kernel_regularizer=regularizers.l2(reg))(Level1_r)
    #output=BatchNormalization(axis=-1)(output)
    output=Lambda(lambda x : softmax(x,axis=-1))(output)
    output=Concatenate(axis=-1)([output,weight_matrix])
    model=Model(inputs=inputs,outputs=output)
    return model

def weighted_loss(y_true,y_pred):
    weight_matrix=K.flatten(y_pred[:,:,:,-1])
    y_pre=y_pred[:,:,:,:-1]
    E=-1/(1)*K.dot(K.transpose(K.expand_dims(weight_matrix,axis=-1)),K.expand_dims(K.log(K.flatten(tf.math.reduce_sum(tf.multiply(y_true,y_pre),-1))),axis=-1))
    return E[:,0]

class DataGenerator(Sequence):
    def __init__(self, path, list_X=list(range(1,4501)), batch_size=20, dim=(432,432), shuffle=True):
        'Initialization'
        self.dim=dim
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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

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
        y = np.empty((self.batch_size, *self.dim, 7))

        # Generate data
        for i, j in enumerate(list_X_temp):
            # Store sample
            arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
            
            X[i,] = np.stack([arr[:,:,0],arr[:,:,-1]],axis=-1)

            # Store class
            y[i,] = arr[:,:,1:-1]

        return X, y


def leg_incremental(modelObj: DynamicDLModel, trainingdata: dict, trainingoutputs):
    from unused.create_train import create_train_leg
    import src.dafne_dl.common.preprocess_train as pretrain
    import os
    import math
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras import optimizers
    try:
        np
    except:
        import numpy as np
    create_train_leg(trainingdata,trainingoutputs)
    path='./train/leg'
    batch_size=5
    card=len(os.listdir(path))
    steps=int(math.floor(float(card/batch_size)))
    pretrain.input_creation(path,card,432,64,7)
    netc = modelObj.model
    checkpoint_path="./Weights_incremental/weights_leg-{epoch:02d}-{loss:.2f}.hdf5" 
    check=ModelCheckpoint(filepath=checkpoint_path, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=5)
    adamlr = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    training_generator = DataGenerator(path=path,list_X=list(range(1,steps*batch_size+1)),batch_size=batch_size)  
    netc.compile(loss=weighted_loss,optimizer=adamlr)
    history=netc.fit_generator(generator=training_generator,steps_per_epoch=steps,epochs=5,callbacks=[check],verbose=1)
    

model = gamba_unet()
model.load_weights('weights/weights_gamba.hdf5')
weights = model.get_weights()

modelObject = DynamicDLModel('ba333b4d-90e7-4108-aca5-9216f408d91e',
                             gamba_unet,
                             incremental_learn_function=leg_incremental,
                             weights = weights
                             )

with open('models/incremental_leg.model', 'wb') as f:
    modelObject.dump(f)
