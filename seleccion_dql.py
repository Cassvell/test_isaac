#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:11:45 2024

@author: isaac
"""
import numpy as np
import kneed as kn
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import splrep, splev
import os
import sys

def get_diurnalvar(data, idx_daily, st):
    ndata = len(data)
    ndays = int(ndata/1440)
    hourly_sample = int(ndata/60) 
    tw = np.linspace(0, hourly_sample-1, hourly_sample)

    #IQR_hr = hourly_IQR(data)                     
    iqr_picks = max_IQR(data, 60, 24)    
    
    qd_baseline = []
#LISTA DE DÍAS QUIETOS LOCALES   
    n = 5 
    qd_list = get_qd_dd(iqr_picks, idx_daily, 'qdl', n)
    qdl = [[0] * 1440 for _ in range(n)]
    #print(data['2024-02-04'])
###############################################################################
#diurnal variation computation
###############################################################################
   # QDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']    
    print('qdl list, \t H[nT] \n')     

    for i in range(n):
        qd = (str(qd_list[i])[0:10])
        qd_arr = data[qd]
        qdl[i] = qd_arr
        qdl[i] = qdl[i].reset_index()
      #  print(qdl[i])
        qdl[i] = qdl[i]['H(nT)']
        qd_2h = qdl[i][0:240]
       # plt.plot(qd_2h)
        qdl[i] = qdl[i]-np.nanmean(qd_2h)
        print(qd, ' | ',  max(qdl[i]))
      #  plt.plot(qdl[i], label=i+': '+qd)
 
    
    # Convert qdl to a numpy array for easy manipulation
    qdl = np.array(qdl)

    # Generate the average array
    qd_baseline = np.nanmedian(qdl, axis=0)
    #plt.plot(qd_baseline, color='k', linewidth=4.0, label = '<QDL>')
    qd_hourly_sample = []

    for i in range(int(len(qd_baseline)/(60))):
    #    print(qd_baseline[i*60:(i+1)*60-1])
        mod = np.nanmedian(qd_baseline[i*60:(i+1)*60-1])
        qd_hourly_sample.append(mod)
    x = np.linspace(0,23,24)
    
    diurnal_baseline = np.tile(qd_hourly_sample, ndays)        

#    print(diurnal_baseline_min)
    kneedle = kn.KneeLocator(
        x,
        qd_hourly_sample,
        S = 1.0,
        curve='convex',
        direction='increasing'#,
        #interp_method='interp1d'
    )
    knee_point = kneedle.knee #elbow_point = kneedle.elbow
    print(f'\n knee point for QDL: {knee_point}')
    
    QD_sample1 = []
    QD_sample2 = np.copy(qd_hourly_sample)
    QD_sample3 = np.copy(qd_hourly_sample)
    

    for i in range(len(qd_hourly_sample)):
        if qd_hourly_sample[i] >= knee_point:
            QD_sample1.append(qd_hourly_sample[i])


    for i in range(len(QD_sample2)):
        if QD_sample2[i] >= knee_point:
            QD_sample2[i] = np.nan
    
    # Set values next to NaNs to NaN
    for i in range(len(QD_sample2)):
        if np.isnan(QD_sample2[i]):
            if i < len(QD_sample2) - 1:  # Set next value to NaN if it exists
                QD_sample2[i + 1] = np.nan        

    for i in range(len(QD_sample3)):
        if QD_sample3[i] >= knee_point:
            QD_sample3[i] = np.nan

    for i in range(len(QD_sample3)):
        if np.isnan(QD_sample3[i]):
            # Propagate NaNs backwards until i = 0
            for j in range(i, -1, -1):
                QD_sample3[j] = np.nan
   

    QD_sample2 = np.array(QD_sample2)
    QD_sample2 = QD_sample2[~np.isnan(QD_sample2)]     
    x2 = np.linspace(0,len(QD_sample2)-1, len(QD_sample2))
    QDH2 = POLY_FIT(x2, QD_sample2, ndegree=4)
    fit2 = QDH2.yfit
   # plt.plot(x2, QD_sample2, 'o')

    QD_sample3 = np.array(QD_sample3)
    QD_sample3 = QD_sample3[~np.isnan(QD_sample3)]     

    QD_sample = [x for n in (QD_sample1,QD_sample3) for x in n]     
    #plt.plot(QD_sample, 'o')

    x1 = np.linspace(0,len(QD_sample)-1, len(QD_sample))
    QDH1 = POLY_FIT(x1, QD_sample, ndegree=5)
    fit1 = QDH1.yfit    
  #  plt.plot(QD_sample1, 'o')
    
    QD = [x for n in (fit2,fit1) for x in n]     
    
    QD_baseline = np.tile(QD, ndays)  
    #agregar un proceso extra para eliminar artefact        
    interpol = splrep(tw, QD_baseline,k=3,s=5)
    
    # Evaluar la interpolación en puntos específicos
    time_axis = np.linspace(min(tw), max(tw), ndata)
    QD_baseline_min = splev(time_axis, interpol)

    mw = 61 #ventana móvil de una hora
    median_filtered = medfilt(QD_baseline_min, kernel_size=mw)
    kernel = np.ones(mw) / mw
    qdl_sm = np.convolve(median_filtered, kernel, mode='same')
    qdl_sm = np.array(qdl_sm)    
    return QD_baseline_min


#generates an array of variation picks    
def max_IQR(data, tw, tw_pick):
    ndata = len(data)
    ndays = int(ndata / 1440)
    if 24 % tw_pick == 0:
        n = 24 / tw_pick
    else:
        print('Please, enter a time window in hours, divisor of 24 h')
        sys.exit()
    
    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / 60)
        
        hourly = []
        for i in range(hourly_sample):
            # Check for NaNs in the current time window
            if not np.all(np.isnan(data[i * tw : (i + 1) * tw])):
                QR1_hr = np.nanquantile(data[i * tw : (i + 1) * tw], .25)
                QR3_hr = np.nanquantile(data[i * tw : (i + 1) * tw], .75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            hourly.append(iqr_hr)
        return hourly
        
    hourly = hourly_IQR(data)
    daily = []
    for i in range(int(n * ndays)):
        iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
        if not np.all(np.isnan(iqr_mov)):
            iqr_picks = np.nanmax(iqr_mov)
            if tw_pick == 24:
                iqr_picks = np.nanmax(iqr_mov)
        else:
            iqr_picks = np.nan
        daily.append(iqr_picks)
        
    return daily
###############################################################################
#based on IQR picks index, select either the 5 QDL in date yyyy-mm-dd format
#in case of type_list = 'qdl' if type_list = I_iqr, it returns a list of the 
#IQR picks per day
#idx_daily: index array containing daily days in format (yyyy-mm-dd)
#type_list puede ser 'qdl' para entregar una lista ordenada de menor a mayor
#para de las fechas con menor IQR.
#type_list = 'I_iqr' es para obtener una lista de todos los días del mes, 
#ordenada en función de la fecha.

def get_qd_dd(data, idx_daily, type_list, n):
    
    daily_var = {'Date': idx_daily, 'VarIndex': data}
    
    local_var = pd.DataFrame(data=daily_var)
    local_var = local_var.sort_values(by = "VarIndex", ignore_index=True)
    
    if type_list == 'qdl':
        local_var = local_var[0:n]['Date']   
    elif type_list == 'I_iqr':
        local_var = local_var.sort_values(by = "Date", ignore_index=True)
    return local_var

picks = max_IQR(data, 60, 6) #los picos al ser de 6 h, implica una muestra 
                             #de ndays X 6 picos   
daily_picks = max_IQR(data, 60, 24)    
                             
n = ndays
list_days = get_qd_dd(daily_picks, idx_daily, 'I_iqr', n) 
i_iqr = list_days['VarIndex'] #ÍNDICES DE IQR
x, GPD, threshold = get_threshold(picks)
#x y GPD son los valores [X,Y] del ajuste GPD. El tercer valor es el umbral que 
#nos interesa.


# Iterate over the daily_stacked array and apply the threshold
for i in range(len(daily_stacked)): #datos de cada día
    if i_iqr[i] >= threshold: #los días donde I_iqr supere el umbral, serán NAN
        i_iqr[i] = np.nan
        daily_stacked[i] = i_iqr[i] #los días donde la fecha coincida con la 
    else:   #fecha del I_iqr mayor que el umbral, harán que todos los valores 
            #de ese día sean NAN            
        daily_stacked[i] = daily_stacked[i]
daily_stacked = np.array(daily_stacked)                                  
###############################################################################
                             
                            