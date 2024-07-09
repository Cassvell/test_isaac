#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:36:23 2024

@author: isaac
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statistics import mode
from scipy.optimize import curve_fit 
import sys
import pandas as pd
from aux_time_DF import index_gen, convert_date
from magnetic_datstruct import get_dataframe



st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
fdate = sys.argv[3]

enddata = fdate+ ' 23:59:00'
idx = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='T')
idx_hr = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='H')    
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')


path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)


def mode_movil(data, mw):
    ###############################################################################
    #cálculo de moda móvil
    ###############################################################################  
    ndata = len(data)
    ndays = int(ndata/1440)
    night_data = ndays*4
    mw_sample = int(ndata/mw) 
    mode_stacked = []      #moda 
    ac_mode = []
    mode_sampled = []
    for i in range(mw_sample):
        mod =  mode(data[i*mw:(i+1)*mw-1])
        mode_sampled.append(mod)
    
    for i in range(mw_sample):
       
        if i == 0:
            # For the first time window, use the first time window and the next time window
            tw_mode = mode_sampled[i:(i+2)]
            ac_mode[i:(i+2)] += tw_mode
        elif i == mw_sample - 1:
            # For the last time window, use the previous time window and the last time window
            tw_mode = mode_sampled[(i-2):(i+1)]
            ac_mode[(i-2):(i+1)] += tw_mode
        else:
            # For all other time window, use the previous time window, the current time window, 
            # and the next time window
            tw_mode = mode_sampled[(i-1):(i+2)]
            ac_mode[(i-1):(i+2)] += tw_mode
        
        sum_mode = np.nanmean(tw_mode)

        mode_stacked.append(sum_mode)

    return mode_stacked

def gaus_center(data, mw):
    
    ndata = len(data)
    mw_sample = int(ndata/mw) 
    avr = np.nanmean(data)
    desv_est = np.nanstd(data)
    ############################################################################### 
    #se calcula un ajuste gaussiano
    #Para el procedimiento del ajuste gausiano, se requiere ausencia de Nans, por lo que se ignoran

    gauss_center = []
    for i in range(mw_sample):
        data_nonan = data[~np.isnan(data)] 
        hist, bin_edges = np.histogram(data_nonan, bins=mw_sample-1, density=True)
            
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
        
        x_interval_for_fit = np.linspace(bin_edges[0], bin_edges[-1], mw_sample-1)
            
        initial_guess = [1, avr, desv_est]
        
        #se define una función para distribución normal
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
        # Ajuste Gaussiano to
        popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
        
        # graficando el ajuste gaussiano
        x_interval_for_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        gaussfit =  gaussian(x_interval_for_fit, *popt)

        gauss_cen_tmp = popt[1]
        gauss_center.append(gauss_cen_tmp)
    
    return gauss_center
   # plt.set_title('Distribución Gausiana', fontsize=18)
    #plt.hist(data_nonan, bins=mw_sample - 1, density=True, alpha=0.6, color='g', label='Data')
    #plt.plot(x_interval_for_fit, gaussfit, color='red', lw=2, label='Fitted Gaussian')
    #plt.axvline(x = popt[1], color = 'k', label = 'center of GF')     
    #plt.grid()
   # plt.set_xlabel('H component distribution [bins = 1 h]')
   # plt.set_ylabel('prob', fontweight='bold')
   # plt.legend()
   # plt.show()

def typical_value(mode, center_gauss, ndata):    
    ###########################################################################################
    #Decide weather the typical value for a day is gonna be either the center of gauss fit or mode
    t_val = []
    for i in range(int(ndata)):
    
        if not center_gauss[i] > mode[i]:
            tmp_val = mode[i]
        else:
            tmp_val = center_gauss[i]    

        t_val.append(tmp_val)
        print(tmp_val)
    return t_val

data = get_dataframe(filenames, path, idx, dates)
#
#tv = typical_value(data)

hourly_mode = mode_movil(data, 60)

daily_gauss = gaus_center(data, 1440)
daily_mode = mode_movil(hourly_mode, 24)
daily_sample = len(data)/1440
typical_dailyval = typical_value(daily_mode, daily_gauss, daily_sample)

