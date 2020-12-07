#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 09:36:01 2020

@author: francesco
"""

from dl.DynamicDLModel import DynamicDLModel
import numpy as np # this is assumed to be available in every context

def gamba_unet():
    
    from keras.layers import Layer, InputSpec
    from keras import initializers, regularizers, constraints
    from keras.activations import softmax
    from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Concatenate, Lambda, ZeroPadding2D, Activation, Reshape, Add
    from keras.models import Sequential, Model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, Callback 
    from keras.utils import plot_model, Sequence

    
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
    #Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=ZeroPadding2D(padding=(1,1))(Level2_l)
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
    #Level3_l=ZeroPadding2D(padding=(1,1))(Level3_l)
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
    #Level4_l=ZeroPadding2D(padding=(1,1))(Level4_l)
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
    #Level5_l=BatchNormalization(axis=-1)(Level5_l) 
    #Level5_l=ZeroPadding2D(padding=(1,1))(Level5_l)
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
    #Level5_l=BatchNormalization(axis=-1)(Level5_l) 
    #Level5_l=ZeroPadding2D(padding=(1,1))(Level5_l)
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
    #Level4_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level5_l)
    #Level4_r=Conv2D(filters=256,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_r)
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
    #Level4_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level5_l)
    #Level4_r=Conv2D(filters=256,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_r)
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
    #Level3_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level4_r)
    #Level3_r=Conv2D(filters=128,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_r)
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
    #Level2_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level3_r)
    #Level2_r=Conv2D(filters=64,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_r)
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
    #Level1_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level2_r)
    #Level1_r=Conv2D(filters=32,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_r)
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

def gamba_apply(modelObj: DynamicDLModel, data: dict):
    from dl.common.padorcut import padorcut
    from scipy.ndimage import zoom
    try:
        np
    except:
        import numpy as np
    
    LABELS_DICT = {
        1: 'SOL',
        2: 'GM',
        3: 'GL',
        4: 'TA',
        5: 'ELD',
        6: 'PE',
        }
    
    MODEL_RESOLUTION = np.array([1.037037, 1.037037])
    MODEL_SIZE = (432, 432)
    netc = modelObj.model
    resolution = np.array(data['resolution'])
    zoomFactor = resolution/MODEL_RESOLUTION
    img = data['image']
    originalShape = img.shape
    img = zoom(img, zoomFactor) # resample the image to the model resolution
    img = padorcut(img, MODEL_SIZE)
    segmentation = netc.predict(np.expand_dims(np.stack([img,np.zeros(MODEL_SIZE)],axis=-1),axis=0))
    labelsMask = np.argmax(np.squeeze(segmentation[0,:,:,:7]), axis=2)
    labelsMask = zoom(labelsMask, 1 / zoomFactor, order=0)
    labelsMask = padorcut(labelsMask, originalShape).astype(np.int8)
    outputLabels = {}
    for labelValue, labelName in LABELS_DICT.items():
        outputLabels[labelName] = (labelsMask == labelValue).astype(np.int8)
    
    return outputLabels


def leg_incremental_mem(modelObj: DynamicDLModel, trainingData: dict, trainingOutputs):
    import dl.common.preprocess_train as pretrain
    from dl.common.DataGenerators import DataGeneratorMem
    import os, time
    from keras.callbacks import ModelCheckpoint
    from keras import optimizers
    try:
        np
    except:
        import numpy as np

    LABELS_DICT = {
        1: 'SOL',
        2: 'GM',
        3: 'GL',
        4: 'TA',
        5: 'ELD',
        6: 'PE',
        }
    MODEL_RESOLUTION = np.array([1.037037, 1.037037])
    MODEL_SIZE = (432, 432)
    BAND = 49
    BATCH_SIZE = 5
    CHECKPOINT_PATH = os.path.join(".", "Weights_incremental", "leg")
    MIN_TRAINING_IMAGES = 5

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    t = time.time()
    print('Image preprocess')

    image_list, mask_list = pretrain.common_input_process(LABELS_DICT, MODEL_RESOLUTION, MODEL_SIZE, trainingData,
                                                          trainingOutputs)

    print('Done. Elapsed', time.time() - t)
    nImages = len(image_list)

    if nImages < MIN_TRAINING_IMAGES:
        print("Not enough images for training")
        return

    print("image shape", image_list[0].shape)
    print("mask shape", mask_list[0].shape)

    print('Weight calculation')
    t = time.time()

    output_data_structure = pretrain.input_creation_mem(image_list, mask_list, BAND)

    print('Done. Elapsed', time.time() - t)

    card = len(image_list)
    steps = int(float(card) / BATCH_SIZE)

    print(f'Incremental learning for leg with {nImages} images')
    t = time.time()

    netc = modelObj.model
    checkpoint_files = os.path.join(CHECKPOINT_PATH, "weights - {epoch: 02d} - {loss: .2f}.hdf5")
    training_generator = DataGeneratorMem(output_data_structure, list_X=list(range(steps * BATCH_SIZE)),
                                          batch_size=BATCH_SIZE, dim=MODEL_SIZE)
    # check = ModelCheckpoint(filepath=checkpoint_files, monitor='loss', verbose=0, save_best_only=False,save_weights_only=True, mode='auto', period=10)
    check = ModelCheckpoint(filepath=checkpoint_files, monitor='loss', verbose=0, save_best_only=False,
                            save_freq='epoch',
                            save_weights_only=True, mode='auto')
    adamlr = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    netc.compile(loss=pretrain.weighted_loss, optimizer=adamlr)
    # history = netc.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=5, callbacks=[check], verbose=1)
    history = netc.fit(x=training_generator, steps_per_epoch=steps, epochs=5, callbacks=[check], verbose=1)
    print('Done. Elapsed', time.time() - t)


model = gamba_unet()
model.load_weights('weights/weights_gamba.hdf5')
weights = model.get_weights()

modelObject = DynamicDLModel('ba333b4d-90e7-4108-aca5-9216f408d91e',
                             gamba_unet,
                             gamba_apply,
                             incremental_learn_function=leg_incremental_mem,
                             weights=weights
                             )

with open('models/leg.model', 'wb') as f:
    modelObject.dump(f)
