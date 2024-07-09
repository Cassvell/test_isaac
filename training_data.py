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
from magdata_processing import get_qd_dd, max_IQR, base_line

###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
###############################################################################
st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
fdate = sys.argv[3]

idate = datetime.strptime(idate + ' 00:00:00', '%Y%m%d %H:%M:%S')
fdate = datetime.strptime(fdate + ' 23:59:00', '%Y%m%d %H:%M:%S')

iw = fdate - timedelta(days=27)-timedelta(hours=23)-timedelta(minutes=59)
###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')                        
fw_dates = []


path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 



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


iqr_picks = max_IQR(data, 60, 24)

baseline, trend, DD = base_line(data, idx, idx_daily)

H_det = data - baseline    
H_det[np.isnan(H_det)] = 999.9   
status_array = np.zeros((len(idx_daily), 27))
data_array = np.zeros((len(idx_daily), 27, 1440))

C = np.zeros((len(idx_daily), 1440))
stat = np.zeros((len(idx_daily), 27))
for i in range(len(idx_daily)):
    fw = idx_daily[i] + timedelta(days=27)
    fw_dates.append(fw)
    
    #idx_daily = idx_daily + timedelta(days=1)
    tmp_index = pd.date_range(start = pd.Timestamp(str(idx_daily[i])), \
                            end = pd.Timestamp(str(fw)), freq='D')
    #print(tmp_index)    
    tmp_inicio = str(idx_daily[i])[0:10]
    tmp_final= str(fw)[0:10]
    tmp_data = H_det[tmp_inicio:tmp_final]
    
    iqr_picks = max_IQR(tmp_data, 60, 24)
    
    list_of_dates = get_qd_dd(iqr_picks, tmp_index, 'qdl', len(tmp_index))
    Ddates = list_of_dates[-6:-1]
    
    days_of_month = np.zeros(len(tmp_index))
    tmp = {'date' : tmp_index, 'status' : days_of_month}
    monthly_status = pd.DataFrame(tmp)
    monthly_status.loc[monthly_status['date'].isin(Ddates), 'status'] = 1

    A = np.array(monthly_status['status'])
    #print(idx_daily[i])
    
    M = [[0] * 1440 for _ in range(27)]
    tmp_data = np.array(tmp_data)
    
    for j in range(27):
        data_array[i, j, :] = tmp_data[j*1440:(j+1)*1440]
        status_array[i, j] = A[j] if j < len(A) else 0  # Ensure no out of range errors
        #print(data_array[i, j, :])
        
    if idx_daily[i] >= iw:
        break
C = np.array(C)
stat = np.array(stat)
print(data_array)
print(status_array)

#np.save('/home/isaac/tools/test/test_isaac/training_data.npy', data_array)
#np.save('/home/isaac/tools/test/test_isaac/status_day.npy', status_array)


#print(M)
#plt.show()