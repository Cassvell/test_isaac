#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:52:48 2024
@author: isaac
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from magnetic_datstruct import get_dataframe
from aux_time_DF import index_gen, convert_date

from datetime import datetime, date, timedelta
from magdata_processing import get_qd_dd, max_IQR

###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
###############################################################################
st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
#fdate = sys.argv[3]

idate = datetime.strptime(idate + ' 00:00:00', '%Y%m%d %H:%M:%S')
fdate = idate + timedelta(days=27) + timedelta(hours=23) + timedelta(minutes=59)

###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')

#filenames = []
path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 
df = pd.read_excel(path_qdl+'qdl.ods', engine='odf', sheet_name=0)


filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)

data = get_dataframe(filenames, path, idx, dates)
###############################################################################
###############################################################################
#GET LOCAL QUIET DAYS!
iqr_picks = max_IQR(data, 60, 24)
n = 10
m = 5
list_qdays = get_qd_dd(iqr_picks, idx_daily, 'qdl', n)# df['enero'] #
qdl = [[0] * 1440 for _ in range(n)]
qdl_sm = [[0] * 1440 for _ in range(n)]
    #print(data['2024-02-04'])
#print(df)
    #print(data['2024-02-04'])
###############################################################################
#diurnal variation computation
###############################################################################
print('qdl list, \t H[nT] \n')   
plt.figure(dpi=200)  

mw = 61
#median_filtered = medfilt(qd_baseline, kernel_size=mw)
kernel = np.ones(mw) / mw

#list_1 = list_qdays[0:5]
#list_2 = list_qdays[6:10]
#plt.figure(figsize=(8,5), dpi=200)
for i in range(n):
    qd = (str(list_qdays[i])[0:10])
    qd_arr = data[qd]
    qdl[i] = qd_arr
    qdl[i] = qdl[i].reset_index()
      #  print(qdl[i])
    qdl[i] = qdl[i]['H(nT)']
    qd_2h = qdl[i][0:240]
       # plt.plot(qd_2h)
    qdl[i] = qdl[i]-np.nanmedian(qd_2h)
    qdl_sm[i] = np.convolve(qdl[i], kernel, mode='same')
    print(qd)
    #print(qd, ' | ',  max(qdl_sm[i]))
    #plt.plot(convolved_result)
   # plt.plot(qdl_sm[i], label='QD'+str(i+1)+': '+qd)    
  #  plt.legend(loc='upper left')
#plt.show() 
print(f'final date: {fdate}')
'''
qdl_sm = np.array(qdl_sm)

qd_baseline = np.nanmean(qdl_sm, axis=0)
#qd_baseline_sm = np.nanmean(qdl, axis=0)

mw = 61
median_filtered = medfilt(qd_baseline, kernel_size=mw)
kernel = np.ones(mw) / mw
qdl_sm = np.convolve(median_filtered, kernel, mode='same')
qdl_sm = np.array(qdl_sm)

#plt.plot(qd_baseline, color='k', linewidth=4.0, label = '<QDL>')
#plt.plot(qd_baseline, color='b', linewidth=4.0, label = '<QDL> sm')
#plt.legend()
plt.show()  
###############################################################################
#diurnal variation computation
###############################################################################
#diurnal_baseline = get_diurnalvar(data, idx_daily, st)
#plt.plot(diurnal_baseline)

#mw = 30
#median_filtered = medfilt(diurnal_baseline, kernel_size=mw)
#kernel = np.ones(mw) / mw
#convolved_result = np.convolve(median_filtered, kernel, mode='same')
#plt.plot(convolved_result)
#plt.show()

'''
