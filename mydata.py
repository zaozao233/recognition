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
        if path.find("seg")==-1 and path.find("mobile")==-1 and path.find('sim') == -1 and path.find('test') == -1 and path.find('labels') == -1:
            path_dir.append(path)
            #print(path)
    
    path_dir = sorted(path_dir)
    #print("sorted")
    #print(path_dir)
    return path_dir

def processing(p_unique):
    data = pd.DataFrame()
    for p in p_unique:
        target = p + "\\sim_dataset.csv"
        df = pd.read_csv(target,header=None)
        data = data.append(df)        
    #print("data:\n",data)
    #data.to_csv('D:\\test\\project2files\\files\\down\\test.csv',header = False,index = False)
    x = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    #print(x)
    #print(y)
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
    y = np.random.randint(3,size=lenth)
    #return pd.Series(y)
    return y

def create_label_series(lenth,label):
    
    y = label*np.ones(lenth)
    return y

def create_labels(p_unique):
    pre_states = []
    pre_act = []
    #print(p_unique)
    for id in p_unique:
        print(id)
        l=id.split('\\')
        pre_states.append(l[0]) 
        pre_act.append(l[1])
    
    state = list(set(pre_states))
    state = sorted(state)
    action = list(set(pre_act)) 
    action = sorted(action)  
    size = len(state)*len(action)
    
    labels = np.arange(size).reshape(len(state),len(action))
    print(labels)
    df = pd.DataFrame(labels,columns=action,index=state)
    print(df) 
    cwd = os.getcwd()
    path = cwd + "/labels.csv"
    df.to_csv(path)
    return df

def select_label(dataframe,path):
    l=path.split('\\')
    state,action = l[0],l[1]
    label = dataframe.loc[state][action]
    return label

def load_data (files):
    #labels = list(set([id.split('\\')[2] for id in files]))
    p = []        
    for file in files:
        (path,t_filename) = os.path.split(file)
        p.append(path)
    p_unique = list(set(p))
    label_df = create_labels(p_unique)
    
    for f in files:               
        #df =  pd.read_csv(f,usecols = [2,3,4],header = None)
        print("-------------->",f)
        label = select_label(label_df,f)
        print(label)
        df = resample_f(f)
        df[2] = df[2].astype('float64')
        df[3] = df[3].astype('float64')
        df[4] = df[4].astype('float64')
        #print("after resampling:\n",df)
        max = df.max()
        min = df.min()
        df_norm = (df - min) / (max - min)
        (temp,t_name) = os.path.split(f)
        # 5 kinds of sensors
        if not path == temp:
            path = temp
            count = 1           
            data_path = temp + '\\sim_dataset.csv'
            df_final = df_norm
            print('_________________new df__________________')
            print(df_final)
            print('_______________end new df________________')
            
        else:
            df_final = pd.concat([df_final,df_norm],axis = 1)
            count+=1
            if count == 5:               
                df_final = df_final.dropna(how='any')                
                #data_y = generate_fake_label(df_final.shape[0])
                data_y = create_label_series(df_final.shape[0],label)
                print(data_y)
                df_final['label'] = pd.Series(data_y,index=df_final.index)
                print(">>>>>>>>>>>>>inserted labels<<<<<<<<<<<<<")
                print(df_final)                
                print('---------------drop it ------------------')
                print(df_final)
                df_final.to_csv(data_path,header = False)
    return p_unique

def resample_f(file):
    #print(file)
    (path,t_filename) = os.path.split(file) 
    (name,extension) = os.path.splitext(t_filename) #split the file path
    new_f = name + ".png"
    new_p = os.path.join(path,new_f)

    df = pd.read_csv(file,usecols = [0,2,3,4],header = None)      
    y = pd.to_datetime(df[0],unit='ms')   
    #print(df)    
    df[0] = y
    
    #plt.figure()
    plt.cla()
    ax = plt.gca()
    df.plot(x=0,y=2,style='b*-',label='origin',ax=ax)
    
    df2 = df.set_index(0).resample('20ms').mean()
    
    df3 = df2.interpolate(method ='linear')
    df = df3 #successfully resampled
     
    df2.reset_index().plot(x=0,y=2,label = "mean resample",color='Red',marker= 'o',ax = ax)
    
    plot_f = df3.plot(y=2,color="Green",label="interpolated",ax = ax)   
    #plt.show()
    fig = plot_f.get_figure()
    fig.savefig(new_p)  
    
           
    return df
  

def preprocess(dir):
    files = []
    p_unique = []
    files = find_dir(dir)
    p_unique = load_data(files)
    target_filename = "C:\\Users\\Zhiha\\Documents\\file_only\\research\\tool_program\\ori\\Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch\\data\\processed\\mygestures.data"
    generate_data(target_filename,p_unique)


if __name__ == '__main__':
    base_path = 'D:\\test\\project2files'
    os.chdir(base_path) #change path
    preprocess(base_path)
    print("done")