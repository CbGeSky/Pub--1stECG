import sys
import os
import numpy as np
import scipy.io as sio
import random
from decimal import Decimal
import argparse
import csv
from  keras.models import load_model
import f_model
from f_preprocess import fill_length

# Usage: python rematch_challenge.py test_file_path

def arg_parse():
    """
    Parse arguements

    """
    parser = argparse.ArgumentParser(description='Rematch test of ECG Contest')
    parser.add_argument("--test_path", dest='test_path', help=
                        "the file path of Test Data",
                        default="your test_path", type=str)

    #You need to write your test data path with the argparse parameter.
    #For your convenience when testing with local data, you can write your local test set path to default


    return parser.parse_args()



def main():

    args = arg_parse()
    test_path = args.test_path
    print(test_path)


    ## Add your codes to  classify normal and diseases.
    model01_path = '/media/uuser/data/final_run/model/model_01.h5'
    model02_path = '/media/uuser/data/final_run/model/model_02.h5'
    modelxg_path = '/media/uuser/data/final_run/model/model.pkl'
    feature_path = '/media/uuser/data/final_run/data/feature.csv'

    keysname = ('I','II','III','aVR','aVL','aVF', \
    'V1','V2','V3','V4','V5','V6','age','sex')

    t_len = 25000
    len_target=t_len
    model_01 = f_model.build_model_01(num_classes=10,len_target=len_target)
    model_01.load_weights(model01_path)

    ##  Classify the samples of the test set and write the results into answers.txt,
    ##  and each row representing a prediction of one sample.
    ##  Here we use random numbers as prediction labels as an example and
    ##  you should replace it with your own results.
    Data_list = os.listdir(test_path)
    classes = np.asarray([1,1,2,3,4,5,6,7,8,9])
    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File_name', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10'])
        for file_name in Data_list:

            if file_name.endswith('.mat'):
                answer = []
                record_name = file_name.strip('.mat')
                answer.append(record_name)
                # model 01
                ecg = np.empty([t_len,12])
                mypath=test_path+file_name
                data = sio.loadmat(mypath)
                # read 12 leads
                for lead in range(12):
                    temp=data[keysname[lead]]
                    ecg[:,lead] = fill_length(temp,t_len)
                data_x = ecg.reshape((1,t_len,12))
                pred_1 = model_01.predict(data_x)
                # model 02
                
                # model xgboost
                                
                preds = pred_1
                preds[preds>=0.5] = 1
                preds[preds<0.5] = 0
                pred_out = preds * classes

                y_out =[]
                for i in range(10):
                    if pred_out[0][i]==classes[i]:
                        y_out.append(i)
                for x in range(10-len(y_out)):
                    y_out.append('')
                writer.writerow(answer+y_out)

        csvfile.close()


if __name__ == "__main__":
    main()
