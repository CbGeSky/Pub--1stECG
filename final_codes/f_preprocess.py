### data generation
### 预处理操作相关
### 主要就是信号长度补齐和训练集数据标签生成的函数
### 后续可以考虑训练过程中改用generator，这样内存需求小

import pandas as pd
import numpy as np
import os
import scipy.io as sio
import random
from sklearn.preprocessing import MultiLabelBinarizer
### read_data
# path here
finalPath = '/media/uuser/data/final_run/'
workPath  = '/media/uuser/data/01_Project/data/'
trainPath = '/media/jdcloud/Train/'
valPath   = '/media/jdcloud/Val/'
ref_name  = '/media/jdcloud/reference.csv'

keysname = ('I','II','III','aVR','aVL','aVF', \
    'V1','V2','V3','V4','V5','V6','age','sex')

def fill_length(ecg,t_len = 50000):
    '''
    信号长度补齐
    '''
    # ecg is a 1-D array
    len_s = len(ecg[0])
    if len_s < t_len:
        len_f = (t_len - len_s) // 2
        return np.pad(np.reshape(ecg,(len_s,)), (len_f,t_len-len_s-len_f),'wrap').T
    else:
        return ecg[0][0:t_len].T 

def data_gen(data_path,id_list,len_target = 50000):
    '''
    single label
    data generation
    '''
    files=os.listdir(data_path)
    files = sorted(files)
    num_records = len(id_list)
    t_len = len_target
    data_x = np.empty([num_records,t_len,12])   
    for i in range(num_records):
    #for f in files:
        ecg = np.empty([t_len,12])
        mypath=data_path+files[id_list[i]]
        data = sio.loadmat(mypath)
        # read 12 leads
        for lead in range(12):
            temp=data[keysname[lead]]
            ecg[:,lead] = fill_length(temp,t_len)
        data_x[i,:,:] = ecg.reshape((1,t_len,12))
    return data_x

def data_gen_12_leads(data_path,ref_path,id_list,len_target = 50000):
    '''
    multi-labels 12 leads
    data and labels generation
    '''
    files=os.listdir(data_path)
    files = sorted(files)
    num_records = len(id_list)
    t_len = len_target
    data_x = np.empty([num_records,t_len,12])   

    f=open(ref_path)
    label=pd.read_csv(f)
    f.close()
    num_records,num_columns = label.shape
    data_y = np.empty([num_records,10,1])
    tag = []
    for i in range(num_records):
        ecg = np.empty([t_len,12])
        mypath=data_path+files[id_list[i]]
        data = sio.loadmat(mypath)

        for lead in range(12):
            temp=data[keysname[lead]]
            ecg[:,lead] = fill_length(temp,t_len)
        data_x[i,:,:] = ecg.reshape((1,t_len,12))
        temp = np.asarray(label.values[i,1:num_columns],dtype='float')
        temp = temp[~np.isnan(temp)]
        tag.append(temp.astype(int).tolist())
    mlb = MultiLabelBinarizer()                                                          
    data_y = mlb.fit_transform(tag)
    return data_x,data_y

def data_gen_18_leads(data_path,frft_path,ref_path,id_list,len_target = 50000):
    '''
    multi-labels 18 leads
    data and labels generation
    '''
    files=os.listdir(data_path)
    files = sorted(files)
    num_records = len(id_list)
    t_len = len_target
    data_x = np.empty([num_records,t_len,18])   

    f=open(ref_path)
    label=pd.read_csv(f)
    f.close()
    num_records,num_columns = label.shape
    data_y = np.empty([num_records,10,1])
    tag = []
    for i in range(num_records):
    #for f in files:
        ecg = np.empty([t_len,18])
        mypath=data_path+files[id_list[i]]
        data = sio.loadmat(mypath)
        mypath=frft_path+files[id_list[i]]
        frft_signal = sio.loadmat(mypath)
        # read 12 leads
        for lead in range(12):
            temp=data[keysname[lead]]
            ecg[:,lead] = fill_length(temp,t_len)
        for lead in range(12,18):
            ecg[:,lead] = frft_signal[:,lead-12]
        data_x[i,:,:] = ecg.reshape((1,t_len,18))
        temp = np.asarray(label.values[i,1:num_columns],dtype='float')
        temp = temp[~np.isnan(temp)]
        tag.append(temp.astype(int).tolist())
    mlb = MultiLabelBinarizer()                                                          
    data_y = mlb.fit_transform(tag)
    return data_x,data_y

def get_index(classes,ref_path=ref_name):
    '''classes count from 0'''
    # dropped not used
    label_name = ('label_0','label_1','label_2','label_3','label_4', \
        'label_5', 'label_6','label_7','label_8')
    labels = sio.loadmat(ref_path)
    idx1 = labels[label_name[classes]].tolist()
    idx2 = []
    for i in range(9):
        if i != classes:
            id = labels[label_name[0]].tolist() 
            random.shuffle(id)
            num = round(len(id) / 7703 * len(idx1))
            id2+=id[0:num]
    return idx1,idx2
