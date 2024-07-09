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
from Ffitting import fourier_series
#from symfit import parameters, variables, sin, cos, Fit

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
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################

imonth = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', \
          '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01']

fmonth = ['2023-01-31', '2023-02-28', '2023-03-31', '2023-04-30', '2023-05-31', '2023-06-30', \
          '2023-07-31', '2023-08-31', '2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31']

#filenames = []
path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 
df = pd.read_excel(path_qdl+'qdl.ods', engine='odf', sheet_name=0)

def nan_array(size):
    return np.full(int(size), np.nan)
def findMiddle(arr):
    l = len(arr)
    if l % 2 == 0:
        # Even length
        return int((arr[l//2 - 1] + arr[l//2]) // 2)
    else:
        # Odd length
        return arr[l//2]

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
ndata = len(data)
###############################################################################
###############################################################################
#GET LOCAL QUIET DAYS per month!
qdl_list_per_month = [[''] * 5 for _ in range(12)]
qdl_per_month = [[0] * 1440 for _ in range(12)]
noon_val = [[0] for _ in range(12)]

season_sample = []

# Iterate over each month (assuming imonth and fmonth are defined and valid)
for i in range(12):
    # Extract data for the current month
    data_per_month = data[imonth[i]:fmonth[i]]
    
    # Calculate the number of days in the month
    days_per_month = len(data_per_month) // 1440  # Assuming data is minute-wise
    days_per_month = int(days_per_month)

    # Proceed only if the current month data is not all NaN
    if not np.all(np.isnan(data_per_month)):
        # Create a date range for the days in the current month
        idx_daily = pd.date_range(start=pd.Timestamp(imonth[i]), 
                                  end=pd.Timestamp(fmonth[i]) + pd.DateOffset(hours=23, minutes=59), freq='D')
        
        # Calculate IQR picks for the current month data
        iqr_picks = max_IQR(data_per_month, 60, 24)
        iqr_picks = np.array(iqr_picks)

        # Filter out NaN values to get valid days
        valid_days = iqr_picks[~np.isnan(iqr_picks)]
        
        # Only proceed if there are more than 9 valid days
        if len(valid_days) > 9:
            qdl_list_per_month[i] = get_qd_dd(iqr_picks, idx_daily, 'qdl', 5)
            qdl = [[0] * 1440 for _ in range(5)]  # Initialize the list for quiet days
            
            for j in range(5):
                qd = str(qdl_list_per_month[i][j])[:10]  # Get the date as string yyyy-mm-dd

                # Extract the data for the quiet day
                qd_arr = data_per_month.loc[qd]
                qdl[j] = qd_arr.reset_index(drop=True)
                
                qd_2h = qdl[j][:240]  # Get the first 2 hours (assuming 1 minute resolution)

                qdl[j] = qdl[j] - np.nanmedian(qd_2h)  # Adjust the data by subtracting the median

            qdl_per_month[i] = np.nanmedian(qdl, axis=0)  # Calculate the median for the month
            noon_val[i] = np.nanmedian(qdl_per_month[i][1019:1079])
        else:   
            noon_val[i] = np.nan

        tmp_monthly_trend = np.full(len(data_per_month) // 60, np.nan)  # Use np.full for NaN initialization
        
        mid_month = len(tmp_monthly_trend) // 2

        tmp_monthly_trend[mid_month] = noon_val[i]

        season_sample.append(tmp_monthly_trend)

season_data = np.concatenate(season_sample)
#print(season_sample)
#qdl_per_month = np.array(qdl_per_month)
x = np.linspace(0, len(season_data)-1, len(season_data))


#plt.plot(x, season_data, 'o')
#plt.show()