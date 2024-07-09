#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:07:44 2024

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

from magdata_processing import base_line
###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
###############################################################################
st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
fdate = sys.argv[3]

###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################
idx = index_gen(idate, fdate, 'T')
idx_daily = index_gen(idate, fdate, 'D')

#filenames = []
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
    

data = get_dataframe(filenames, path, idx, dates)

#monthly base line
#base_line = base_line(data, idx, idx_daily) 
