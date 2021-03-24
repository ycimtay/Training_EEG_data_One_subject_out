import pandas as pd
import numpy as np
import hdf5storage
from keras.layers import GaussianNoise
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.applications import InceptionResNetV2
from keras.applications import DenseNet201
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
import scipy.io as sio
from keras.regularizers import l2
from numpy import empty
import pandas as pd
import cv2
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#config = tf.ConfigProto(gpu_options=gpu_options)
#sess = tf.Session(config=config)

#datapoints hold the starting and ending line number of data of each participant in overall data(dataAll)
datapoints = np.load("datapoints.npy")
dataAll = np.load("dataAll.npy")
labelAll = np.load("labelAll.npy")

datapoints = np.squeeze(datapoints)
datapoints[0] = 0

#One subject out- Training

for j in range(10,11):
    out_subject = j

    sum = 0
    n1 = 0
    n2 = 0
    dataTest = []

    #dataTest holds the data of subject-out
    dataTest = dataAll[int(datapoints[j]):int(datapoints[j+1]),:,:,:]




    ind = np.delete(range(0,dataAll.shape[0]),range(int(datapoints[j]),int(datapoints[j+1])))
    dataTrain = []
    dataTrain = dataAll[(ind),:,:,:]

    labelTest = labelAll[int(datapoints[j]):int(datapoints[j+1]),:]
    y_true = np.argmax(labelTest, axis=1)


    labelTrain = labelAll[(ind),:]



    xtrain, xval, ytrain, yval = train_test_split(dataTrain, labelTrain, test_size=0.5, shuffle=True)


    # parameters
    batch_size = 64
    num_epochs = 1
    verbose = 1
    num_classes = 3
    patience = 150
    base_path = 'C:\\models'

    l2_regularization = 0.01

    # data generator
    aug = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        zoom_range=0,
        horizontal_flip=False)

    # model parameters
    regularization = l2(l2_regularization)

    base_model=InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(80,125,3)) #imports the mobilenet model and discards the last 1000 neuron layer.

    x=base_model.output

    x=GlobalAveragePooling2D()(x)
    #x= GaussianNoise(0.1)(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x = GaussianNoise(0.1)(x)
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x = GaussianNoise(0.1)(x)
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x = GaussianNoise(0.1)(x)
    #x=Dense(1024,activation='relu')(x) #dense layer 2
    x = GaussianNoise(0.1)(x)
    x=Dense(512,activation='relu')(x) #dense layer 3
    x = GaussianNoise(0.1)(x)
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


    log_file_path = base_path + str(out_subject)+'emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 8), verbose=1)
    trained_models_path = base_path + str(out_subject)+'_inceptionresnetDeap'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, reduce_lr, early_stop]
    #model.fit(x=xtrain, y=ytrain, epochs=num_epochs,validation_data=(xtest,ytest), batch_size=64, verbose=1,shuffle=True, steps_per_epoch=None, callbacks=callbacks)
    model.fit_generator(aug.flow(xtrain, ytrain, batch_size=64), validation_data=(xval,yval),shuffle=True,steps_per_epoch=len(dataTest) // 64,epochs=100,callbacks=callbacks)





