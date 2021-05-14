import numpy as np
import csv
import pandas as pd
import os
import glob

from datetime import datetime
import time 
import matplotlib.pyplot as plt
import random
import statistics as stc

""" a=np.array([[[1,2,3],[4,5,6]],
            [[7,8,9],[10,11,12]],
            [[13,14,15],[16,17,18]]])
unpermuted=torch.tensor(a)
print(unpermuted.size())
permuted=unpermuted.permute(0,2,1)
print(permuted.size())
print(permuted) """

def find_dir():
    path_dir = []
    print("not sorted")
    for path in glob.iglob('**/*.txt',recursive=True):
        if path.find("seg")==-1 and path.find("watch")==-1 and path.find('sim') == -1 and path.find('labels') == -1 :
            (p,t_filename) = os.path.split(path)
            
            path_dir.append(p)                
    path_dir = list(set(sorted(path_dir)))
    return path_dir

def gettime_txt(path):
    with open(path,'r') as p:
        l = p.readline().split(',')
        start_r,start_s,end_r,end_s = l[0],l[1],l[2],l[3]
    return  int(start_r),int(start_s),int(end_r),int(end_s)   

def processing(paths):
    """ p = []        
    for file in files:
        (path,t_filename) = os.path.split(file)
        p.append(path)
    p_unique = list(set(p))

    for f in p_unique:
        m_start_r,m_start_s,m_end_r,m_end_s = gettime_txt(f+'\\mobile_log.txt')
        w_start_r,w_start_s,w_end_r,w_end_s = gettime_txt(f+'\\watch_log.txt') """
    dlt_send_m_s = []
    dlt_send_m_e = []
    dlt_rsp_w_s = []
    dlt_rsp_w_e = []

    dlt_str_w = []
    dlt_end_w = []
    dlt_str_m = []
    dlt_end_m = []

    dlt_timelapse_s = []
    dlt_timelapse_e = []
    delay_s = []
    delay_e = []

    for p in paths:
        m_start_r,m_start_s,m_end_r,m_end_s = gettime_txt(p+'\\mobile_log.txt')
        w_start_r,w_start_s,w_end_r,w_end_s = gettime_txt(p+'\\watch_log.txt')
        m_start = []
        m_end = []
        w_start = []
        w_end = []
        for file in glob.iglob(p+'\\*.csv',recursive=False):
            if file.find('sim') == -1:
                #print(file)
                df = pd.read_csv(file,usecols = [0,1],header = None)            
                if not file.find("_mobile")==-1:                
                    m_start.append(int(df.iloc[0,0]))
                    m_end.append(int(df.iloc[-2,0]))
                elif not file.find("_watch")==-1:
                    w_start.append(int(df.iloc[0,0]))
                    w_end.append(int(df.iloc[-1,0]))
        ###try to evaluate the time difference
        dlt_send_m_s.append(m_start_s - m_start_r)
        dlt_send_m_e.append(m_end_s - m_end_r)
        dlt_rsp_w_s.append(w_start_r - w_start_s)
        dlt_rsp_w_e.append(w_end_r - w_end_s)
        #print('>>>>>>>>>>>>>>>>>>>>>>--------1')
        dlt_str_w.append(min(w_start) - w_start_r) 
        dlt_end_w.append(max(w_end) - w_end_s) 
        dlt_str_m.append(min(m_start) - m_start_r) 
        dlt_end_m.append(max(m_end) - m_end_s) 
        #print('>>>>>>>>>>>>>>>>>>>>>>--------2')
        dlt_timelapse_s.append(m_start_s - w_start_r)
        dlt_timelapse_e.append(m_end_s - w_end_r)
        delay_s.append((m_start_s - m_start_r-(w_start_r - w_start_s))/2)
        delay_e.append((m_end_s - m_end_r-(w_start_r - w_start_s))/2)

    x = np.arange(len(paths))
    ax1 = plt.subplot(231)
    ax1.plot(x,dlt_send_m_s,color = 'green', label = ' △sent_mobile(start)')
    ax1.plot(x,dlt_send_m_e,color = 'red', label = ' △sent_mobile(end)')
    plt.legend()
    ###
    ax2 = plt.subplot(232)
    ax2.plot(x,dlt_rsp_w_s,color = 'black', label = ' △response_watch(start)')
    ax2.plot(x,dlt_rsp_w_e,color = 'blue', label = ' △response_watch(end)')
    plt.legend()
    ax3 = plt.subplot(233)
    ax3.plot(x,dlt_str_w,color = 'green', label = ' time var for watch(start)')
    ax3.plot(x,dlt_end_w,color = 'red', label = ' time var for watch(end)')
    plt.legend()
    ax4 = plt.subplot(234)
    ax4.plot(x,dlt_str_m,color = 'black', label = ' time var for mobile(start)')
    ax4.plot(x,dlt_end_m,color = 'blue', label = ' time var for mobile(end)')
    plt.legend()
    ax5 = plt.subplot(235)
    ax5.plot(x,dlt_timelapse_s,color = 'green', label = ' time var for sending(start)')
    ax5.plot(x,dlt_timelapse_e,color = 'red', label = ' time var for sending(end)')
    plt.legend()
    ax6 = plt.subplot(236)
    ax6.plot(x,delay_s,color = 'green', label = ' delay for sending(start)')
    ax6.plot(x,delay_e,color = 'red', label = ' delay for sending(end)')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    base_path = 'D:\\test\\project2files\\2\\test\\files\\standing\\down\\'
    os.chdir(base_path) #change path
    dir = find_dir()
    print(dir)
    processing(dir)





