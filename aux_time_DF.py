#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:34:25 2024

@author: isaac
"""
import pandas as pd
from datetime import datetime

def convert_date(input_date, input_format, output_format):
    # Convert input string to datetime object
    date_obj = datetime.strptime(input_date, input_format)
    # Convert datetime object to desired string format
    formatted_date = date_obj.strftime(output_format)
    return formatted_date

def index_gen(idate, fdate, res):
    idate = convert_date(idate, '%Y%m%d', '%Y-%m-%d')
    fdate = convert_date(fdate, '%Y%m%d', '%Y-%m-%d')
    
    if res == 'T':
        enddata = fdate + ' 23:59:00'
    elif res == 'D':
        enddata = fdate
    elif res == 'H':
        enddata = fdate + ' 23:00:00'
    
    idx = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq=res)
    return idx        
    
def list_names(daterange, string1, string2):
    select_fnames = []
    for i in daterange:
        tmp_name = string1+str(i)+string2
        select_fnames.append(tmp_name)
    return(select_fnames)