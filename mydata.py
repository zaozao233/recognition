import numpy as np
import csv
import pandas as pd
import os
import glob
import _pickle as cp
from datetime import datetime
import time 
import matplotlib.pyplot as plt

NB_SENSOR_CHANNELS = 15
""" def normalization (data,max,min):
    
    _range = max - min
    data = (data - min) / _range
    print(max)
    print(min)
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data """

def find_dir(dir):
    path_dir = []
    print("not sorted")
    for path in glob.iglob('**/*.csv',recursive=True):
        if path.find("seg")==-1 and path.find("mobile")==-1 and path.find('sim') == -1 and path.find('test') == -1:
            path_dir.append(path)
            print(path)
    
    path_dir = sorted(path_dir)
    print("sorted")
    print(path_dir)
    return path_dir

def processing(p_unique):
    data = pd.DataFrame()
    for p in p_unique:
        target = p + "\\sim_dataset.csv"
        df = pd.read_csv(target,header=None)
        data = data.append(df)
        print("append one data sample")
    print("data:\n",data)
    #data.to_csv('D:\\test\\project2files\\files\\down\\test.csv',header = False,index = False)
    x = data.iloc[:,1:]
    y = data.iloc[:,0]
    
    data_x = x.to_numpy()
    data_y = y.to_numpy()
    
    return data_x,data_y

def generate_data(target_filename,p_unique):
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))
    data_x,data_y=processing(p_unique)
    nb_samples = data_x.shape[0]
    nb_training_samples = int(0.7*nb_samples)
    X_train,Y_train = data_x[:nb_training_samples,:],data_y[:nb_training_samples]
    X_test,Y_test = data_x[nb_training_samples:,:],data_y[nb_training_samples:]

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    obj = [(X_train,Y_train),(X_test, Y_test)]
    f = open(target_filename, 'wb')
    cp.dump(obj, f, protocol=-1)
    f.close()

def generate_fake_label(lenth):
    x = list(range(1,lenth))
    y = list()
    for item in x:
        if item % 7 == 0:
            y.append(1)
        else:
            y.append(0)
    return pd.Series(y)

def load_data (files):
    #labels = list(set([id.split('\\')[2] for id in files]))
    p = []        
    for file in files:
        (path,t_filename) = os.path.split(file)
        p.append(path)
    p_unique = list(set(p))
    print("p_unique")
    print(p_unique)
    """ for f in p_unique:
        data_y = np.random.randint(0,2,len(df_norm))
        data = f + "\\" + "sim_dataset.csv" """

    for f in files:               
        #df =  pd.read_csv(f,usecols = [2,3,4],header = None)
        df = resample_f(f)
        df[2] = df[2].astype('float64')
        df[3] = df[3].astype('float64')
        df[4] = df[4].astype('float64')

        max = df.max()
        min = df.min()
        print(f)
        print(max)
        print(min)
        df_norm = (df - min) / (max - min)
        print(df_norm)
        #data_x = df_norm.to_numpy()
        (temp,t_name) = os.path.split(f)
        # 5 kinds of sensors
        if not path == temp:
            path = temp
            count = 1
            #data_y = np.random.randint(0,2,len(df_norm))
            data_y = generate_fake_label(len(df_norm))
            data_path = temp + '\\sim_dataset.csv'
            print("--------------------------\ndata_y", len(data_y))
            #with open(data_path,'w')
            df_norm.insert(0,'label',data_y)
            df_final = df_norm
            print('_______________new df__________________')
            print(df_final)
            print('_____________end new df__________________')
            
        else:
            df_final = pd.concat([df_final,df_norm],axis = 1)
            count+=1
            if count == 5:
                print(df_final)
                df_final = df_final.dropna(how='any')
                print('------------drop it ---------------')
                print(df_final)
                df_final.to_csv(data_path,header = False,index = False)
    return p_unique

def resample_f(file):
    print(file)
    df = pd.read_csv(file,usecols = [0,2,3,4],header = None)  
    y = pd.to_datetime(df[0],unit='ms')   
    print(df)
    df.plot.scatter(x=0,y=4)
    df[0] = y
    #df2 = df.set_index(0).resample('20ms').interpolate('linear')
    df2 = df.set_index(0).resample('20ms').mean()
    df3 = df2.interpolate(method ='linear')
    print(df2) 
    df2.plot(y=4)
    df3.plot(y=4)
    df = df3 #successfully resampled
    """ df3.plot(x='0',y='2')
    df.plot(x='0',y='2') """
    plt.show()
    #df2.to_csv('D:\\test\\project2files\\files\\left\\20210206103331721\\test.csv', header = None )
    return df

    
    


def preprocess(dir):
    files = []
    p_unique = []
    files = find_dir(dir)
    p_unique = load_data(files)
    target_filename = "C:\\Users\\Zhiha\\Documents\\file_only\\research\\tool_program\\ori\\Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch\\data\\processed\\mygestures.data"
    generate_data(target_filename,p_unique)


if __name__ == '__main__':
    base_path = 'D:\\test\\project2files\\sitting\\down'
    os.chdir(base_path) #change path
    #resample_f('D:\\test\\project2files\\walking\\null\\20210329103941937\\grav_watch.csv')
    preprocess(base_path)