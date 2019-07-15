# CIFAR-100

The aim of this project is to classify Images labelled over 100 categories.We will be using Convolutional Neural Networks to perform the training on dataset CIFAR 100.

Tools used : keras 2.1.1,numpy,matplotlib
# Importing the necessary libraries :
Note : use keras 2.1.1 , It can be installed as follows:

                         !pip install keras==2.1.1
                                  
Import all the required libraries to build the model as follows:
                                  
                         from __future__ import print_function
                         import keras
                         from keras.datasets import cifar100
                         from keras.models import Sequential
                         from keras.layers import Dense, Dropout, Activation, Flatten
                         from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
                         from keras.optimizers import SGD
                         from keras.regularizers import l2
                         from keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint
                         from keras.preprocessing.image import ImageDataGenerator
                         from keras.utils import print_summary, to_categorical
                         from keras import backend as K
                         import sys
                         import os
                         import numpy as np
                         import matplotlib.pyplot as plt

# Data Loading :
CIFAR data sets are one of the most well-known data sets in computer vision tasks created by Geoffrey Hinton, Alex Krizhevsky and Vinod Nair.There are 100 different category labels containing 600 images for each (1 testing image for 5 training images per class). The 100 classes in the CIFAR-100 are grouped into 20 super-classes. Each image comes with a “fine” label (the class to which it belongs) and a “coarse” label (the super-class to which it belongs). We will work with the fine labels.
![classes](https://user-images.githubusercontent.com/49706927/61202818-3c034380-a706-11e9-836e-0c6a4785c5dd.png)

The Dataset can be loaded from keras as follows:

                      from keras.datasets import cifar100
                      (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

Dataset consists 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

dimensions of training and testing data set can be printed with the code:
                       
                       print(x_train.shape,x_test.shape)
                       print(y_train.shape,y_test.shape)

x_train :(50000,32,32,3) ,
x_test  :(10000,32,32,3) , 
y_train :(50000,1) ,
y_test  :(10000,1)

# Feature Extraction and building model using Convolutional Neural Network :
First few Images can be printed as follows :
 
                       m=4
                       k=0
                       plt.figure(1)
                       for i in range(0,m):
                           for j in range(0,m):
                                plt.subplot2grid((m,m),(i,j))
                                plt.imshow(x_train[k])
                                k=k+1
                                plt.show()

![op](https://user-images.githubusercontent.com/49706927/61197733-153c1180-a6f4-11e9-881c-ba32dfd0aed3.png)

Now,Initialise the variables used to train our model like Batch size,number of classes,number of epochs,Dropout rate,learning rate,decay rate,model path,momentum rate as follows:


                       BATCH_SIZE = 100
                       NUM_CLASSES = 100
                       EPOCHS = 165000
                       INIT_DROPOUT_RATE = 0.5
                       MOMENTUM_RATE = 0.9
                       INIT_LEARNING_RATE = 0.01
                       L2_DECAY_RATE = 0.0005
                       CROP_SIZE = 32
                       LOG_DIR = './logs'
                       MODEL_PATH = './keras_cifar100_model.h5'

Now,To perform the Classification on dataset,we need to perform One Hot Encode with keras as below:
Why one Hot Encoding ??

A one hot encoding is a representation of categorical variables as binary vectors.This first requires that the categorical values be mapped to integer values.Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.

The Keras library offers a function called to_categorical() that can be used to one hot encode integer data.

                       y_train = to_categorical(y_train, NUM_CLASSES)
                       y_test = to_categorical(y_test, NUM_CLASSES)
                       
                       print(y_train.shape,y_test.shape)
y_train :(50000,100)
y_test  :(10000,100)

Now,for convenience to deal with data,Let's Normalise the Images:

                       x_train = x_train.astype('float32')
                       x_test = x_test.astype('float32')
                       x_train /= 255.0
                       x_test /= 255.0
                       
The dataset can be preprocessed with global contrast normalization (sample-wise centering) and ZCA whitening. Additionally, the images are padded with four 0 pixels at all borders (2D zero padding layer at the top of the model). The model should be trained 32x32 random crops with random horizontal flipping. That’s all for data augmentation.

# Network : The CNN Architecture
18 convolutional layers are arranged in stacks of
(layers x units x receptive fields)
([1×384×3],[1×384×1,1×384×2,2×640×2],[1×640×1,3×768×2],[1×768×1,2×896×2],[1×896×3,2×1024×2],[1×1024×1,1×1152×2],[1×1152×1],[1×100×1])
                       
                       model = Sequential()
                       model.add(ZeroPadding2D(4, input_shape=x_train.shape[1:]))
                       
                       # Stack 1:
                       model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))

                       # Stack 2:
                       model.add(Conv2D(384, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(384, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))

                       # Stack 3:
                       model.add(Conv2D(640, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))

                       # Stack 4:
                       model.add(Conv2D(768, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))

                       # Stack 5:
                       model.add(Conv2D(896, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))

                       # Stack 6:
                       model.add(Conv2D(1024, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Conv2D(1152, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))

                       # Stack 7:
                       model.add(Conv2D(1152, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
                       model.add(Activation('elu'))
                       model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
                       model.add(Dropout(INIT_DROPOUT_RATE))
                       model.add(Flatten())
                       model.add(Dense(NUM_CLASSES))
                       model.add(Activation('softmax'))
                     
Here,Advanced Activation Layers from keras is used namely, ELU (Exponential Linear Unit).
ELUs have exponential term in the formula and the derivative of an exponential term equals to the exponential term itself. For the forward propagation, all weights and biases are activated with some constant multiplication of an exponential of them, and they are back-propagated with the derivative of the activation function, it is actually exponential of all weights and biases. The formula can be seen below.

The exponential linear unit (ELU) with 0 < α is
f(x) =  x              if   x > 0 and 
f(x) =  α(exp(x)−1)    if   x ≤ 0 

The main problem in ReLUs is that their mean activation is not zero. In the other words, ReLUs are not zero-centered activation functions and it leads to shift the bias term for the term in the next layer. But then, ELUs arranges the mean of the activation closer to zero because they have negative values, even if these values are very close to zero, and it converges faster, it means the model will learn faster.

![elu vs relu](https://user-images.githubusercontent.com/49706927/61196780-998b9600-a6ee-11e9-9479-6099fb8a7006.png)

Summary of the model can be printed using :
                         
                        model.summary()
 Output :
                     
                       Layer (type)                 Output Shape              Param #   
                       =================================================================
                       zero_padding2d_1 (ZeroPaddin (None, 40, 40, 3)         0         
                       _________________________________________________________________
                       conv2d_1 (Conv2D)            (None, 40, 40, 384)       10752     
                       _________________________________________________________________
                       activation_1 (Activation)    (None, 40, 40, 384)       0         
                       _________________________________________________________________
                       max_pooling2d_1 (MaxPooling2 (None, 20, 20, 384)       0         
                       _________________________________________________________________
                       dropout_1 (Dropout)          (None, 20, 20, 384)       0         
                       _________________________________________________________________
                       conv2d_2 (Conv2D)            (None, 20, 20, 384)       147840    
                       _________________________________________________________________
                       conv2d_3 (Conv2D)            (None, 20, 20, 384)       590208    
                       _________________________________________________________________
                       conv2d_4 (Conv2D)            (None, 20, 20, 640)       983680    
                       _________________________________________________________________
                       conv2d_5 (Conv2D)            (None, 20, 20, 640)       1639040   
                       _________________________________________________________________
                       activation_2 (Activation)    (None, 20, 20, 640)       0         
                       _________________________________________________________________
                       max_pooling2d_2 (MaxPooling2 (None, 10, 10, 640)       0         
                       _________________________________________________________________
                       dropout_2 (Dropout)          (None, 10, 10, 640)       0         
                       _________________________________________________________________
                       conv2d_6 (Conv2D)            (None, 10, 10, 640)       3687040   
                       _________________________________________________________________
                       conv2d_7 (Conv2D)            (None, 10, 10, 768)       1966848   
                       _________________________________________________________________
                       conv2d_8 (Conv2D)            (None, 10, 10, 768)       2360064   
                       _________________________________________________________________
                       conv2d_9 (Conv2D)            (None, 10, 10, 768)       2360064   
                       _________________________________________________________________
                       activation_3 (Activation)    (None, 10, 10, 768)       0         
                       _________________________________________________________________
                       max_pooling2d_3 (MaxPooling2 (None, 5, 5, 768)         0         
                       _________________________________________________________________
                       dropout_3 (Dropout)          (None, 5, 5, 768)         0         
                       _________________________________________________________________
                       conv2d_10 (Conv2D)           (None, 5, 5, 768)         590592    
                       _________________________________________________________________
                       conv2d_11 (Conv2D)           (None, 5, 5, 896)         2753408   
                       _________________________________________________________________
                       conv2d_12 (Conv2D)           (None, 5, 5, 896)         3212160   
                       _________________________________________________________________
                       activation_4 (Activation)    (None, 5, 5, 896)         0         
                       _________________________________________________________________
                       max_pooling2d_4 (MaxPooling2 (None, 3, 3, 896)         0         
                       _________________________________________________________________
                       dropout_4 (Dropout)          (None, 3, 3, 896)         0         
                       _________________________________________________________________
                       conv2d_13 (Conv2D)           (None, 3, 3, 896)         7226240   
                       _________________________________________________________________
                       conv2d_14 (Conv2D)           (None, 3, 3, 1024)        3671040   
                       _________________________________________________________________
                       conv2d_15 (Conv2D)           (None, 3, 3, 1024)        4195328   
                       _________________________________________________________________
                       activation_5 (Activation)    (None, 3, 3, 1024)        0         
                       _________________________________________________________________
                       max_pooling2d_5 (MaxPooling2 (None, 2, 2, 1024)        0         
                       _________________________________________________________________
                       dropout_5 (Dropout)          (None, 2, 2, 1024)        0         
                       _________________________________________________________________
                       conv2d_16 (Conv2D)           (None, 2, 2, 1024)        1049600   
                       _________________________________________________________________
                       conv2d_17 (Conv2D)           (None, 2, 2, 1152)        4719744   
                       _________________________________________________________________
                       activation_6 (Activation)    (None, 2, 2, 1152)        0         
                       _________________________________________________________________
                       max_pooling2d_6 (MaxPooling2 (None, 1, 1, 1152)        0         
                       _________________________________________________________________
                       dropout_6 (Dropout)          (None, 1, 1, 1152)        0         
                       _________________________________________________________________
                       conv2d_18 (Conv2D)           (None, 1, 1, 1152)        1328256   
                       _________________________________________________________________
                       activation_7 (Activation)    (None, 1, 1, 1152)        0         
                       _________________________________________________________________
                       max_pooling2d_7 (MaxPooling2 (None, 1, 1, 1152)        0         
                       _________________________________________________________________
                       dropout_7 (Dropout)          (None, 1, 1, 1152)        0         
                       _________________________________________________________________
                       flatten_1 (Flatten)          (None, 1152)              0         
                       _________________________________________________________________
                       dense_1 (Dense)              (None, 100)               115300    
                       _________________________________________________________________
                       activation_8 (Activation)    (None, 100)               0         
                       =================================================================
                       Total params: 42,607,204
                       Trainable params: 42,607,204
                       Non-trainable params: 0
                       _________________________________________________________________


Now,defining functions to perform adjustments to the model :

For Learning Rate : 
Learning Rate will be decreased by a factor of 10 after 35 iterations.
                 
                 def lr_scheduler(epoch, lr, step_decay = 0.1):
                       return float(lr * step_decay) if epoch == 35.000 else lr
                       
For the drop-out rate:
For the later 50 iterations, the drop-out rate will be increased for all layers in a stack to (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0).

                 def dr_scheduler(epoch, layers, rate_list = [0.0, .1, .2, .3, .4, .5, 0.0], rate_factor = 1.5):
                         if epoch == 85000:
                            for i, layer in enumerate([l for l in layers if "dropout" in np.str.lower(l.name)]):
                                  layer.rate = layer.rate + rate_list[i]
                         elif epoch == 135000:
                              for i, layer in enumerate([l for l in layers if "dropout" in np.str.lower(l.name)]):
                                    layer.rate = layer.rate + layer.rate * rate_factor if layer.rate <= 0.66 else 1
                                    return layers
                                    
 Defining custom callback objects for the learning rate :
 For the last 40 iterations, the drop-out rate will be increased by a factor of 1.5 for all layers.
 
                 class StepLearningRateSchedulerAt(LearningRateScheduler):
                        def __init__(self, schedule, verbose = 0): 
                             super(LearningRateScheduler, self).__init__()
                             self.schedule = schedule
                             self.verbose = verbose
    
                        def on_epoch_begin(self, epoch, logs=None): 
                              if not hasattr(self.model.optimizer, 'lr'):
                                  raise ValueError('Optimizer must have a "lr" attribute.')
            
                              lr = float(K.get_value(self.model.optimizer.lr))
                              lr = self.schedule(epoch, lr)
       
                              if not isinstance(lr, (float, np.float32, np.float64)):
                                   raise ValueError('The output of the "schedule" function ' 'should be float.')
        
                              K.set_value(self.model.optimizer.lr, lr)
                              if self.verbose > 0: 
                                      print('\nEpoch %05d: LearningRateScheduler reducing learning ' 'rate to %s.' % (epoch + 1, lr))
                                      
  For Dropout Rate :
                       
                       class DropoutRateScheduler(Callback):
                            def __init__(self, schedule, verbose = 0):
                                 super(Callback, self).__init__()
                                 self.schedule = schedule
                                 self.verbose = verbose
         
                            def on_epoch_begin(self, epoch, logs=None):
                                if not hasattr(self.model, 'layers'):
                                    raise ValueError('Model must have a "layers" attribute.')
            
                                layers = self.model.layers
                                layers = self.schedule(epoch, layers)
        
                                if not isinstance(layers, list):
                                    raise ValueError('The output of the "schedule" function should be list.')
        
                                self.model.layers = layers
        
                                if self.verbose > 0:
                                    for layer in [l for l in self.model.layers if "dropout" in np.str.lower(l.name)]:
                                print('\nEpoch %05d: Dropout rate for layer %s: %s.' % (epoch + 1, layer.name, layer.rate))
  Defining Random Crop :                             
                     
                         def random_crop(img, random_crop_size):
                              height, width = img.shape[0], img.shape[1]
                              dy, dx = random_crop_size
                              x = np.random.randint(0, width - dx + 1)
                              y = np.random.randint(0, height - dy + 1)
                              return img[y:(y+dy), x:(x+dx), :]
                                
Defining crop generator function :

                         def crop_generator(batches, crop_length, num_channel = 3):
                                while True:
                                     batch_x, batch_y = next(batches)
                                     batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, num_channel))
                                     for i in range(batch_x.shape[0]):
                                     batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
                                     yield (batch_crops, batch_y)

Using optimizer stochastic Gradient Descent,      

                         opt = SGD(lr=INIT_LEARNING_RATE, momentum=MOMENTUM_RATE)
                
                         lr_rate_scheduler = StepLearningRateSchedulerAt(lr_scheduler)
                         dropout_scheduler = DropoutRateScheduler(dr_scheduler)
                         tensorboard = TensorBoard(log_dir=LOG_DIR, batch_size=BATCH_SIZE)
                         checkpointer = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True)
 
compiling the model,where the metrics used are accuracy and top_k_categorical_accuracy and loss = categorical_crossentropy as we need to do  mutlicalss classification                        
                          
                          model.compile(optimizer=opt,
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy', 'top_k_categorical_accuracy'])
# Data Augmentation :                                       
                          datagen = ImageDataGenerator(samplewise_center=True,
                                                       zca_whitening=True,
                                                       horizontal_flip=True
                                                       )
 Fitting the training data : 
 
                          datagen.fit(x_train)
                          
                          train_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
                          train_flow_w_crops = crop_generator(train_flow, CROP_SIZE)
                          valid_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
                          
Finally,Running the epochs :

                          model.fit_generator(train_flow_w_crops,
                                              epochs=EPOCHS,
                                              steps_per_epoch=len(x_train) / BATCH_SIZE,
                                              callbacks=[lr_rate_scheduler, dropout_scheduler, tensorboard, checkpointer],
                                              validation_data=valid_flow,
                                              validation_steps=(len(x_train) / BATCH_SIZE))
                                              
We are done with the training now,as the number of epochs are 16500 it takes a lot of time to run.


Contributed by : KUSHAL SHARMA , SRIPRIYA ARABALA

  
