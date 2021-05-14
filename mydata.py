import numpy as np
import csv
import pandas as pd
import os
import glob
import _pickle as cp
from datetime import datetime
import time 
import matplotlib.pyplot as plt
import random
import statistics as stc

NB_SENSOR_CHANNELS = 15
TRAINING_PERCENTAGE = 0.8

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
        if path.find("seg")==-1 and path.find("mobile")==-1 and path.find('sim') == -1 and path.find('labels') == -1 and path.find('sitting')== -1  :
            #if path.find("Back")==-1  :
            path_dir.append(path)
            #print(path)
    
    path_dir = sorted(path_dir)
    #print("sorted")
    #print(path_dir)
    return path_dir

def processing(paths):
    data = pd.DataFrame()
    temp = []
    for p in paths:
        target = p + "\\sim_dataset.csv"
        if os.path.exists(target):
            df = pd.read_csv(target,header=None)
            data = data.append(df)
        else:
            temp.append(p)

    x = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    #print(x)
    #print(y)
    data_x = x.to_numpy()
    data_y = y.to_numpy()
    print("not exists files: \n",temp)
    
    return data_x,data_y

def shuffle_data(state,action,target_filename):
    
    training_f = []
    testing_f =[]
    NB_SAMPLES = [[]]
    t1 = 0
    
    for s in state:
        for a in action : 
            p =  os.path.join(base_path,s,a)
            t_f =[]
            totalDir = 0 
            with os.scandir(p) as entries:
                for entry in entries:
                    if entry.is_dir():
                        totalDir += 1
                        #print(entry.name)
                        t_f.append(os.path.join(base_path,s,a,entry.name))
            #NB_SAMPLES[t1].append(totalDir)
            #training_f[t1].append(t_f) 
            BP=int(TRAINING_PERCENTAGE*totalDir)
            training_f.extend(t_f[:BP])
            testing_f.extend(t_f[BP:])
        t1 += 1
    """ print(testing_f)
    print(len(testing_f))
    print(len(training_f)) """
    random.shuffle(training_f)
    random.shuffle(testing_f)

    X_train,Y_train = processing(training_f)
    X_test,Y_test = processing(testing_f)
    
    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    obj = [(X_train,Y_train),(X_test, Y_test)]
    f = open(target_filename, 'wb')
    cp.dump(obj, f, protocol=-1)
    f.close()

""" def generate_data(target_filename,p_unique):
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
    f.close() """



def create_label_series(lenth,label):
    
    y = label*np.ones(lenth)
    return y

def create_labels(p_unique):
    pre_states = []
    pre_act = []
    #print(p_unique)
    for id in p_unique:
        #print(id)
        l=id.split('\\')
        pre_states.append(l[0]) 
        pre_act.append(l[1])
    
    state = list(set(pre_states))
    state = sorted(state)
    action = list(set(pre_act)) 
    action = sorted(action)  
    size = len(state)*len(action)
    
    labels = np.arange(size).reshape(len(state),len(action))
    #print(labels)
    df = pd.DataFrame(labels,columns=action,index=state)
    print(df) 
    cwd = os.getcwd()
    path = cwd + "/labels.csv"
    df.to_csv(path)
    return df,state,action

def select_label(dataframe,path):
    l=path.split('\\')
    state,action = l[0],l[1]
    label = dataframe.loc[state][action]
    return label

def drawCumulativeHist(lengths):
    plt.hist(lengths, bins = 20, facecolor= "blue",edgecolor="black", alpha=0.7)
    plt.xlabel('Lengths')
    plt.ylabel('Frequency')
    plt.title('Lengths Of Samples')
    plt.show()

def load_data (files):
    #labels = list(set([id.split('\\')[2] for id in files]))
    p = []        
    for file in files:
        (path,t_filename) = os.path.split(file)
        p.append(path)
    p_unique = list(set(p))
    label_df,state,action = create_labels(p_unique)

    lengths = []
    
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
            
        else:
            df_final = pd.concat([df_final,df_norm],axis = 1)
            count+=1
            if count == 5:               
                df_final = df_final.dropna(how='any')
                #print("old shape: ",df_final.shape[0])
                """ drop_nb =  int(df_final.shape[0] * 0.15)  
                df_final.drop(df_final.tail(drop_nb).index,inplace=True)
                df_final.drop(df_final.head(drop_nb).index,inplace=True) """
                """ plt.cla()
                df_final.plot() """
                df_final.drop(df_final.tail(25).index,inplace=True)
                df_final.drop(df_final.head(5).index,inplace=True)
                """ df_final.plot()
                plt.show() """
                if df_final.empty:
                    print('DataFrame is empty!')
                    print(f)
                    break
                print("new shape: ",df_final.shape[0])
                #data_y = generate_fake_label(df_final.shape[0])
                data_y = create_label_series(df_final.shape[0],label)
                #print(data_y)
                df_final['label'] = pd.Series(data_y,index=df_final.index)
                """ print(">>>>>>>>>>>>>inserted labels<<<<<<<<<<<<<")
                print(df_final)                
                print('---------------drop it ------------------') """
                lengths.append(df_final.shape[0])
                df_final.to_csv(data_path,header = False)
    return state,action,lengths

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
    """ plt.cla()
    ax = plt.gca()
    df.plot(x=0,y=2,style='b*-',label='origin',ax=ax) """
    
    df2 = df.set_index(0).resample('20ms').mean()
    
    df3 = df2.interpolate(method ='linear')
    df = df3 #successfully resampled
     
    """ df2.reset_index().plot(x=0,y=2,label = "mean resample",color='Red',marker= 'o',ax = ax)
    
    plot_f = df3.plot(y=2,color="Green",label="interpolated",ax = ax)   
    #plt.show()
    fig = plot_f.get_figure()
    fig.savefig(new_p) """  
    
           
    return df
  

def preprocess(dir):
    files = []
    p_unique = []
    files = find_dir(dir)
    state,action,lengths = load_data(files)
    target_filename = "C:\\Users\\Zhiha\\Documents\\file_only\\research\\tool_program\\ori\\Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch\\data\\processed\\mygestures_ww.data"
    #generate_data(target_filename,p_unique)
    shuffle_data(state,action,target_filename)
    drawCumulativeHist(lengths)
    b = stc.mean(lengths)
    print(b*20/1000)


if __name__ == '__main__':
    base_path = 'D:\\test\\project2files\\2\\test\\files\\'
    os.chdir(base_path) #change path
    preprocess(base_path)
    print("done")
    