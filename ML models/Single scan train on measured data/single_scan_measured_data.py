#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras import layers, callbacks

from keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

from tensorflow.python.keras.utils.vis_utils import plot_model
import pydot

from scipy.stats import norm
from scipy import stats
import os,fnmatch

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import dataframe_image as dfi
import pickle
import math

from keras.utils.layer_utils import count_params


# In[2]:


sfs = [4, 8,32]
layer_options = [2]
weight_options = [0, 1] #options
extra_options = ['b']
nr_classes = 3
channel_width = 79


# ### Gotta load in some data

# In[3]:


# datapath = '../../../All generated data/'
# labelpath = '../../../All generated labels/'
# data_list = os.listdir(datapath)
# #print(data_list)

data_list = fnmatch.filter(os.listdir('real_data_singlescan'), '*samples.csv')
label_list = fnmatch.filter(os.listdir('real_data_singlescan'), '*labels.csv')

dataFull = []
labelFull = []
entire_data = []

# Preprocessing on the data

for data in data_list:
    dataFull.append(pd.read_csv(os.path.join('real_data_singlescan', data), sep=','))
    
for label in label_list:
    labelFull.append(pd.read_csv(os.path.join('real_data_singlescan', label), sep=','))
    
entire_data = [dataFull[0].values]
entire_labels = [labelFull[0].values]

# Create a list to store the arrays
all_datapoint = []
all_labels = []

for i in range(len(entire_data[0])):

        all_datapoint.append(entire_data[0][i])
        all_labels.append(entire_labels[0][i].reshape((1,79)))
        
        
        
complete_data = []
complete_labels = []

lower_channel = math.floor(channel_width/2)
upper_channel = math.ceil(channel_width/2)

chosen_channels = list(range(upper_channel+1,79-upper_channel,channel_width))

# check if channels are viable
for channel in chosen_channels:
        if (channel - lower_channel) < 0 or (channel + upper_channel - 1) > total_channels:
            print('Bad channel choice')
            exit()
        if channel_width % 2 == 0:
            print('please pick uneven channel width')
            exit()

            
for iter in range(len(all_datapoint)):
   
        complete_data.append(all_datapoint[iter])
        complete_labels.append(all_labels[iter])
    

#quick check to make sure it works
print(complete_data[1].shape)
print(len(complete_data))
print(complete_labels[1].shape)
print(len(complete_labels))


# In[4]:


data_train, data_test, labels_train, labels_test = train_test_split(complete_data, complete_labels, train_size=0.8, random_state=112)

# One hot encoding
#labels_test = to_categorical(labels_test)
#labels_train = to_categorical(labels_train)

data_train = np.array(data_train)
data_test = np.array(data_test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
print(labels_test.shape)


# In[7]:


time_data_amount=1

# Make a scaler from training data
test = data_train[0]

#reshape to 1d features
nr_data_train = data_train.shape[0]
# data_train = data_train.reshape(nr_data_train, time_data_amount*channel_width)
nr_data_test = data_test.shape[0]
# data_test = data_test.reshape(nr_data_test, time_data_amount*channel_width)

scaler = preprocessing.StandardScaler().fit(data_train)

# scale everything using that scaler
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

#reshaping back to 2d features
data_train = data_train.reshape(nr_data_train, channel_width, time_data_amount)
data_test = data_test.reshape(nr_data_test,  channel_width, time_data_amount)


# In[8]:


#fix shape
labels_test = labels_test.reshape(nr_data_test,channel_width)
labels_train = labels_train.reshape(nr_data_train,channel_width)
# data_train = data_train
# data_test = data_test
print(labels_test.shape)
print(labels_train.shape)
print(data_test.shape)
#print(labels_test[1])


labels_test = to_categorical(labels_test)
labels_train = to_categorical(labels_train)
print(labels_test.shape)
#print(labels_test.reshape(920,3,79).shape)
#print(labels_test[1])


# In[9]:


def weighted_mean_squared_error(class_weight):
  def loss(y_true, y_pred):
          y_true = tf.dtypes.cast(y_true, tf.float32)
          y_pred = tf.dtypes.cast(y_pred, tf.float32)
#             y_pred=  tf.transpose(y_pred, perm=[1, 0,2])

          
          weight = tf.constant(class_weight, dtype=tf.float32)
          weight_per_sample = tf.transpose(tf.gather(weight, tf.argmax(y_true, axis=-1)))
          weight_per_sample = tf.expand_dims(weight_per_sample, axis=-1)
#             losses = tf.keras.losses.mean_squared_error(y_true, y_pred)
          losses = tf.math.square(y_true-y_pred)*weight_per_sample
          return tf.reduce_mean(losses, axis=-1)
  return loss


# #### Pick out one channel for each sample
# For now it takes the same channel for all samples

# In[10]:


for layerOpt in layer_options:
    for extraOpt in extra_options:
        for weightOpt in weight_options:
            for sf in sfs:
                if weightOpt == 0:
                    using_weights = False
                else:
                    using_weights = True

                name = 'multi_channels_test' #name of model - should be descriptive
                save_folder = 'sf'+ str(sf) + '_layerOpt' + str(layerOpt) + extraOpt  #hyperparameter description her
                if using_weights == True:
                    save_folder = save_folder + '_W'
                    name = name + '_weighted'
                    if weightOpt == 2:
                        save_folder = save_folder + 'crazier'


                if weightOpt == 2:
                    class_weights = [1.0, 5.0, 36.0]
                else: 
                    class_weights = [1.0, 5.0, 10.0]

                class_weights = np.array([class_weights[i] for i in range(len(class_weights))])

                signal_size = 1

                y = layers.Input(shape=(channel_width,1), dtype='float32', name='Input')
                x = layers.Conv1D(sf, 3, padding='same', activation='relu', use_bias=True)(y)
                for iter in range(layerOpt-1):
                    x = layers.Conv1D(sf, 3, padding='same', activation='relu')(x)

                x = layers.MaxPool1D(pool_size=2,strides=2)(x)
                for iter in range(layerOpt):
                    x = layers.Conv1D(sf*2, 3, padding='same', activation='relu')(x)

                x = layers.MaxPool1D(pool_size=2,strides=2)(x)
                for iter in range(layerOpt):
                    x = layers.Conv1D(sf*4, 3, padding='same', activation='relu')(x)
                #x = layers.Conv1D(sf*4, 3, padding='same', activation='relu')(x)
                x = layers.MaxPool1D(pool_size=2,strides=2)(x)
                
                if extraOpt == 'b': 
                    for iter in range(layerOpt):
                        x = layers.Conv1D(sf*8, 3, padding='same', activation='relu')(x)
                    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
                
                x = layers.Flatten()(x)
                


                class_layer = [{}]*channel_width
                output_layer = [{}]*channel_width
                for iter in range(channel_width):
                    class_layer[iter] = layers.Dropout(rate=0.3)(x)
                    if sf == 2:
                        class_layer[iter] = layers.Dense(3,activation='relu')(class_layer[iter])
                    else: 
                        class_layer[iter] = layers.Dense(sf,activation='relu')(class_layer[iter])
                    class_layer[iter] = layers.Dropout(rate=0.3)(class_layer[iter])
                    #class_layer[iter] = layers.Dense(64,activation='relu')(class_layer[iter])
                    output_layer[iter] = layers.Dense(nr_classes, activation='softmax', name=('out'+str(iter)))(class_layer[iter])



                model = Model(inputs=[y], outputs=[out_layer for out_layer in output_layer])


                isExist = os.path.exists(save_folder)
                if not isExist:
                    os.makedirs(save_folder)
                    print('Created "' + save_folder + '" directory')
                else:
                    print('"'+ save_folder + '" directory already existed - skipping')
                    continue


                nr_params = count_params(model.trainable_weights)

                cre = open(save_folder + '/' + 'Params_' + str(nr_params), 'x')


                # ------------- model compilation --------------
                ourAdam = Adam()
                optimizer = tf.keras.optimizers.RMSprop()

                if using_weights == True:
                        loss_func = weighted_mean_squared_error(class_weights)
                else:
                        loss_func = 'categorical_crossentropy'

                metric_T = 'accuracy'
                loss_dict = {}
                metric_dict = {}
                for iter in range(channel_width):
                        loss_dict['out'+str(iter)] = loss_func
                        metric_dict['out'+str(iter)] = metric_T



                model.compile(optimizer=ourAdam, loss = loss_dict,
                              metrics=metric_T) 


                if nr_params > 1000000:
                    ting = 1
                elif nr_params > 500000:
                    ting = 2
                elif nr_params > 250000:
                    ting = 4    
                elif nr_params > 125000:
                    ting = 8
                elif nr_params < 50000:
                    ting = 32
                elif nr_params < 10000:
                    ting = 64
                else:
                    ting = 16


                BATCH_SIZE = 192*ting
                EPOCH = 600 + 75 * ting
                
                if ting > 15:
                    ekstra = 15
                elif ting < 4:
                    ekstra = 0
                else:
                    ekstra = ting

                # Set the model training parameters
                # Stop model training when the training loss is not dropped
                callbacks_list = [callbacks.EarlyStopping(
                                        monitor='val_loss', 
                                        patience=math.floor(10 + ekstra), 
                                        verbose=0, 
                                        mode='auto',
                                        restore_best_weights=True,
                                    )
                                            ]

                # ------------- Starting model Training --------------

                hist = model.fit(data_train,[labels_train[:,iter,:] for iter in range(channel_width)],
                          batch_size = BATCH_SIZE, 
                          epochs = EPOCH, 
                          callbacks= callbacks_list,
                          verbose=0,
                          validation_split=0.25)


                # Show loss curves
                fig1 = plt.figure()
                plt.title('Training loss')
                plt.plot(hist.epoch, hist.history['loss'], label='train loss')
                plt.plot(hist.epoch, hist.history['val_loss'], label='val_loss')
                plt.legend()
                plt.savefig(save_folder + '/%s Training loss.pdf' %(name), format='pdf')
                #plt.show()
                plt.close(fig1)

                fig2 = plt.figure()
                plt.title('Training accuracy')
                plt.xlabel("Epoch #")
                plt.ylabel("Accuracy")

                lossNames = ['out0_accuracy', 'out'+str(math.floor(channel_width/3))+'_accuracy', 'out'+str(math.floor(channel_width*2/3))+'_accuracy']
                for (i, l) in enumerate(lossNames):
                    # plot the loss for both the training and validation data
                    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
                    plt.plot(hist.epoch, hist.history[l], label=l)
                    plt.plot(hist.epoch, hist.history["val_" + l],
                        label="val_" + l)
                    plt.legend()
                plt.savefig(save_folder + '/%s Training acc.pdf' %(name), format='pdf')
                #plt.show()
                plt.close(fig2)

                evalDict = model.evaluate(data_test,[labels_test[:,iter,:] for iter in range(channel_width)])
                totalA = 0
                for i in range(channel_width+1,channel_width+1+channel_width):
                    totalA += evalDict[i]

                totalA /= channel_width
                print(totalA)

                # Saving dict of history and evaluation result
                with open(save_folder + '/' + 'histDict', 'wb') as file_pi:
                    pickle.dump(hist.history, file_pi)

                with open(save_folder + '/' + 'evalDict' + str(totalA), 'wb') as file_pi:
                    pickle.dump(evalDict, file_pi)

                model.save(save_folder + '/' + name + '_Model')

                #Test on test data
                true_test_labels = np.argmax(labels_test, axis=-1)
                test_predictions = model.predict(data_test)
                test_result = np.argmax(test_predictions, axis=-1).T

                #classification report
                class_names = ['Empty channel', 'Wi-Fi', 'Bluetooth']
                class_report = classification_report(true_test_labels.flatten(), test_result.flatten(),target_names=class_names)

                with open(save_folder + '/' + 'classReportString', 'wb') as file_pi:
                    pickle.dump(class_report, file_pi)

                #Confusion matric plot
                plt.figure()
                ConfusionMatrixDisplay.from_predictions(true_test_labels.flatten(), test_result.flatten(),normalize='true',cmap='Greens',colorbar=False,display_labels=class_names)
                plt.title('Confusion Matrix')
                plt.savefig(save_folder +'/confusion_matrix_'+ name +'.pdf', format='pdf')
                plt.close()

                class_reportDict = classification_report(true_test_labels.flatten(), test_result.flatten(),output_dict=True, target_names=class_names)
                for key in class_reportDict:
                    try:
                        class_reportDict[key]['Samples'] = class_reportDict[key].pop('support')
                    except Exception as e:
                        print(e)
                #print(class_reportDict)
                class_reportDict.pop('accuracy')
                df = pd.DataFrame(class_reportDict).transpose().round(decimals=3)
                dfi.export(df, save_folder + '/' +name + "_ClassReport.png", table_conversion="matplotlib")

