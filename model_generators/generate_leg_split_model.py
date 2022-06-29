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
import shutil
import sys

try:
    from dafne_dl import DynamicDLModel
except ModuleNotFoundError:
    from dl import DynamicDLModel

def gamba_unet():
    
    from tensorflow.keras import regularizers
    from tensorflow.keras.activations import softmax
    from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Concatenate, Lambda, Activation, Reshape, Add
    from tensorflow.keras.models import Model

    inputs=Input(shape=(216,216,2))
    weight_matrix=Lambda(lambda z: z[:,:,:,1])(inputs)
    weight_matrix=Reshape((216,216,1))(weight_matrix)
    reshape=Lambda(lambda z : z[:,:,:,0])(inputs)
    reshape=Reshape((216,216,1))(reshape)

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
    Level6_l=Activation('relu')(Level6_l)
    
    Level5_r=Conv2DTranspose(filters=512,kernel_size=(3,3),strides=3,output_padding=(1,1),kernel_regularizer=regularizers.l2(reg))(Level6_l)
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

    
    Level4_r=Conv2DTranspose(filters=256,kernel_size=(2,2),strides=2,output_padding=(1,1),kernel_regularizer=regularizers.l2(reg))(Level5_r)
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

def gamba_apply(modelObj: DynamicDLModel, data: dict):
    try:
        from dafne_dl.common.padorcut import padorcut
        from dafne_dl.common import biascorrection
        from dafne_dl.common.preprocess_train import split_mirror
        from dafne_dl.labels.leg import long_labels as LABELS_DICT
    except ModuleNotFoundError:
        from dl.common.padorcut import padorcut
        from dl.common import biascorrection
        from dl.common.preprocess_train import split_mirror
        from dl.labels.leg import long_labels as LABELS_DICT

    from scipy.ndimage import zoom
    try:
        np
    except:
        import numpy as np
    
    MODEL_RESOLUTION = np.array([1.037037, 1.037037])
    MODEL_SIZE = (432, 432)
    MODEL_SIZE_SPLIT = (216, 216)

    classification = data.get('classification', '')

    single_side = False
    swap = False

    # Note: this is anatomical right, which means it's image left! It's the image right that is swapped
    if classification.lower().strip().endswith('right'):
        single_side = True
        swap = False
    elif classification.lower().strip().endswith('left'):
        single_side = True
        swap = True

    # otherwise: double sided

    netc = modelObj.model
    resolution = np.array(data['resolution'])
    zoomFactor = resolution / MODEL_RESOLUTION
    img = data['image']
    originalShape = img.shape
    img = zoom(img, zoomFactor)  # resample the image to the model resolution

    if single_side:
        img = padorcut(img, MODEL_SIZE_SPLIT)
        if swap:
            img = img[::1, ::-1]

        segmentation = netc.predict(np.expand_dims(np.stack([img, np.zeros(MODEL_SIZE_SPLIT)], axis=-1), axis=0))
        label = np.argmax(np.squeeze(segmentation[0, :, :, :13]), axis=2)
        if swap:
            label = label[::1, ::-1]

        labelsMask = zoom(label, 1 / zoomFactor, order=0)
        labelsMask = padorcut(labelsMask, originalShape).astype(np.int8)

        outputLabels = {}

        suffix = ''
        if data['split_laterality']:
            suffix = '_L' if swap else '_R'
        for labelValue, labelName in LABELS_DICT.items():
            outputLabels[labelName + suffix] = (labelsMask == labelValue).astype(
                np.int8)  # left in the image is right in the anatomy
        return outputLabels

    # two sides
    img = padorcut(img, MODEL_SIZE)
    imgbc= biascorrection.biascorrection_image(img)
    a1,a2,a3,a4,b1,b2=split_mirror(imgbc)
    left=imgbc[int(b1):int(b2),int(a1):int(a2)]
    left=padorcut(left, MODEL_SIZE_SPLIT)
    right=imgbc[int(b1):int(b2),int(a3):int(a4)]
    right=right[::1,::-1]
    right=padorcut(right, MODEL_SIZE_SPLIT)
    segmentationleft=netc.predict(np.expand_dims(np.stack([left,np.zeros(MODEL_SIZE_SPLIT)],axis=-1),axis=0))
    labelleft=np.argmax(np.squeeze(segmentationleft[0,:,:,:7]), axis=2)
    segmentationright=netc.predict(np.expand_dims(np.stack([right,np.zeros(MODEL_SIZE_SPLIT)],axis=-1),axis=0))
    labelright=np.argmax(np.squeeze(segmentationright[0,:,:,:7]), axis=2)
    labelright=labelright[::1,::-1]
    labelsMask_left=np.zeros(MODEL_SIZE,dtype='float32')
    labelsMask_right=np.zeros(MODEL_SIZE,dtype='float32')
    labelsMask_left[int(b1):int(b2),int(a1):int(a2)]=padorcut(labelleft, [b2-b1, a2-a1])
    labelsMask_right[int(b1):int(b2),int(a3):int(a4)]=padorcut(labelright, [b2-b1, a4-a3])
    labelsMask_left = zoom(labelsMask_left, 1/zoomFactor, order=0)
    labelsMask_left = padorcut(labelsMask_left, originalShape).astype(np.int8)
    labelsMask_right = zoom(labelsMask_right, 1/zoomFactor, order=0)
    labelsMask_right = padorcut(labelsMask_right, originalShape).astype(np.int8)
    outputLabels = {}

    if data['split_laterality']:
        for labelValue, labelName in LABELS_DICT.items():
            outputLabels[labelName+'_R'] = (labelsMask_left == labelValue).astype(np.int8)
            outputLabels[labelName+'_L'] = (labelsMask_right == labelValue).astype(np.int8)
        return outputLabels
    else:
        for labelValue, labelName in LABELS_DICT.items():
            outputLabels[labelName] = np.logical_or(labelsMask_left == labelValue, labelsMask_right == labelValue).astype(np.int8)
        return outputLabels


def leg_incremental_mem(modelObj: DynamicDLModel, trainingData: dict, trainingOutputs,
                        bs=5, minTrainImages=5):
    try:
        import dafne_dl.common.preprocess_train as pretrain
        from dafne_dl.common.DataGenerators import DataGeneratorMem
        from dafne_dl.labels.leg import inverse_labels
    except ModuleNotFoundError:
        import dl.common.preprocess_train as pretrain
        from dl.common.DataGenerators import DataGeneratorMem
        from dl.labels.leg import inverse_labels


    import time
    #from keras.callbacks import ModelCheckpoint
    from tensorflow.keras import optimizers
    try:
        np
    except:
        import numpy as np

    MODEL_RESOLUTION = np.array([1.037037, 1.037037])
    MODEL_SIZE = (432, 432)
    MODEL_SIZE_SPLIT = (216, 216)
    BAND = 64
    BATCH_SIZE = bs
    #CHECKPOINT_PATH = os.path.join(".", "Weights_incremental_split", "leg")
    MIN_TRAINING_IMAGES = minTrainImages

    #os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    t = time.time()
    print('Image preprocess')
    classification = trainingData.get('classification', '')

    single_side = False
    swap = False

    # Note: this is anatomical right, which means it's image left! It's the image right that is swapped
    if classification.lower().strip().endswith('right'):
        single_side = True
        swap = False
    elif classification.lower().strip().endswith('left'):
        single_side = True
        swap = True

    if single_side:
        image_list, mask_list = pretrain.common_input_process_single(inverse_labels, MODEL_RESOLUTION, MODEL_SIZE,
                                                                     MODEL_SIZE_SPLIT, trainingData,
                                                                     trainingOutputs, swap)
    else:
        image_list, mask_list = pretrain.common_input_process_split(inverse_labels, MODEL_RESOLUTION, MODEL_SIZE,
                                                                    MODEL_SIZE_SPLIT, trainingData,
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
    #checkpoint_files = os.path.join(CHECKPOINT_PATH, "weights - {epoch: 02d} - {loss: .2f}.hdf5")
    training_generator = DataGeneratorMem(output_data_structure, list_X=list(range(steps * BATCH_SIZE)),batch_size=BATCH_SIZE, dim=MODEL_SIZE_SPLIT)
    # check = ModelCheckpoint(filepath=checkpoint_files, monitor='loss', verbose=0, save_best_only=False,save_weights_only=True, mode='auto', period=10)
    #check = ModelCheckpoint(filepath=checkpoint_files, monitor='loss', verbose=0, save_best_only=True, # save_freq='epoch',
    #                        save_weights_only=True, mode='auto')
    adamlr = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)
    netc.compile(loss=pretrain.weighted_loss, optimizer=adamlr)
    # history = netc.fit_generator(generator=training_generator, steps_per_epoch=steps, epochs=5, callbacks=[check], verbose=1)
    #history = netc.fit(x=training_generator, steps_per_epoch=steps, epochs=5, callbacks=[check], verbose=1)
    history = netc.fit(x=training_generator, steps_per_epoch=steps, epochs=5, verbose=1)
    print('Done. Elapsed', time.time() - t)


if len(sys.argv) > 1:
    # convert an existing model
    print("Converting model", sys.argv[1])
    old_model_path = sys.argv[1]
    filename = old_model_path
    old_model = DynamicDLModel.Load(open(old_model_path, 'rb'))
    shutil.move(old_model_path, old_model_path + '.bak')
    weights = old_model.get_weights()
    timestamp = old_model.timestamp_id
    model_id = old_model.model_id
else:
    model_id = 'ba333b4d-90e7-4108-aca5-9216f408d91e'
    timestamp = 1610001000
    model = gamba_unet()
    model.load_weights('weights/weights_gamba_split.hdf5')
    weights = model.get_weights()
    filename = f'models/Leg_{timestamp}.model'

modelObject = DynamicDLModel(model_id,
                             gamba_unet,
                             gamba_apply,
                             incremental_learn_function=leg_incremental_mem,
                             weights=weights,
                             timestamp_id=timestamp
                             )

with open(filename, 'wb') as f:
    modelObject.dump(f)

print('Saved', filename)
