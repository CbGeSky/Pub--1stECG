### f1_train.py model 01 训练函数主体
import scipy.io as sio
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.utils import np_utils
from keras import initializers, regularizers, constraints,optimizers
import f_model
from f_preprocess import data_gen_12_leads

### data path 各种路径的设定
workPath  = '/media/uuser/data/01_Project/data/'
trainPath = workPath + '/Train/'
valPath   = workPath + '/Val/'
refPath   = workPath + '/reference.csv'
save_weights_path = '/media/uuser/data/01_Project/model_params/model/'
checkpoint_path = '/media/uuser/data/01_Project/model_params/checkpoint/model_01_'
keysname = ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','age','sex')
data_path = trainPath
ref_name  = refPath

### 用于调整不平衡的类别权重参数
class_weights = {0:1. , 1:4.3 ,  2:4.2 , 3:2.2 , 4:12.5 , 5:3.3 , 6:3.3, 
    7:9.3 , 8:1. , 9: 5.1 }

len_target  = 25000
num_records = 6689

### 数据、标签的生成
### 后续这里的data_gen可以去改用generator，现在这种一次性生成所有数据的方式太吃内存了
X,y = data_gen_12_leads(data_path,ref_name,range(num_records),len_target=len_target)
num_records = X.shape[0]
num_class = 10     
index = np.asarray(range(num_records))
np.random.seed(3666)
np.random.shuffle(index)
x_train = X[index,:,:]
y_train = y[index,:]
#x_train = x_train.astype("float32")
#y_train = y_train.astype("float32")
print('New y_train shape: ', y_train.shape)

model_m = f_model.build_model_01(num_classes=num_class,len_target=len_target)
print(model_m.summary())

### 这里的callback设定的是在checkpointpath下保存当前loss最小的一次
### 会保存很多模型，loss每降一次就会存一个，后期可修改这个位置
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path+'.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
]

model_m.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 50
EPOCHS = 100

history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.33,
                      shuffle=True,
                      class_weight=class_weights,
                      verbose=1)

print('-------------------')
print('Training Finished !')
print('-------------------')
# 权重保存
#history.save_weights( save_weights_path + "weight-saved.")
#history.save( save_weights_path + "model.model.")

### 下面实现了一个评价指标计算的函数
### 要求y是one-hot编码形式，multi-hot for 多类别 
def metrics(y_target,y_pred):
    TP = np.sum( (y_target==1) & (y_pred==1),axis=0)
    TN = np.sum( (y_target==0) & (y_pred==0) ,axis=0)
    FP = np.sum( (y_target==0) & (y_pred==1) ,axis=0)
    FN = np.sum( (y_target==1) & (y_pred==0) ,axis=0)
    return TP,TN,FP,FN

preds = model_m.predict(x_train)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
y_pred = preds
TP,TN,FP,FN = metrics(y_train,y_pred)
precision = TP/(TP+FP)
recall    = TP/(TP+FN)
f1        = 2*precision*recall/(precision+recall)
print('F1_score in training set:')
print(F1,np.mean(F1))
print('Precision in training set:')
print(precision,np.mean(precision))
print('Recall in training set:')
print(recall,np.mean(recall))
print('f1_score in training set:')
print(f1,np.mean(f1))

