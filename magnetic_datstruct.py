#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:28:00 2024

@author: isaac
"""
import pandas as pd
import os
import numpy as np
def get_dataframe(filenames, path, idx, daily_idx):       
    dfs_c = []
    filenames = sorted(filenames)
    for i in range(len(filenames)):
    
        # Construct the full path
        full_path = os.path.join(path, filenames[i])
        
        # Check if the file exists
        if os.path.isfile(full_path):
            try:
                # Read the file into a dataframe
                df_c = pd.read_csv(full_path, header=1, delim_whitespace=True)
                dfs_c.append(df_c)
            except Exception as e:
                # Handle any read errors
                print(f"Error reading {full_path}: {e}")
        else:
            print(f"File not found: {full_path}")
            dailyfile = pd.date_range(start = pd.Timestamp(str(daily_idx[i])), \
                            periods=1440, freq='T')
            df_c = np.empty((1440, 10), dtype=object)
            
            df_c[:, 0] = dailyfile.day.astype(str).str.zfill(2)       # Day (DD)
            df_c[:, 1] = dailyfile.month.astype(str).str.zfill(2)     # Month (MM)
            df_c[:, 2] = dailyfile.year.astype(str)                   # Year (YYYY)
            df_c[:, 3] = dailyfile.hour.astype(str).str.zfill(2)      # Hour (HH)
            df_c[:, 4] = dailyfile.minute.astype(str).str.zfill(2)    # Minute (MM)
            #Fill the last 5 columns with np.nan
            df_c[:, 5:] = np.nan               
            df_c = pd.DataFrame(df_c)                
    
    # Concatenate all dataframes in the list
    if dfs_c:
        final_df = pd.concat(dfs_c, ignore_index=True)
        print("Dataframe concatenated successfully.")
    else:
        print("No valid dataframes to concatenate.")
     #   else:
###############################################################################
#COMBINE THE DAILY DATAFRAMES        
       # dfs_c.append(df_c) 
        
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    
    #df = df.replace(999999.0, np.NaN)        
    
    dd = df.iloc[:,0].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    mm = df.iloc[:,1].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    yy = df.iloc[:,2].astype(str)
    
    df['dt_tmp']= yy+mm+dd
    
    #print(df)
    df['date']  = pd.to_datetime(df['dt_tmp'], format='%Y-%m-%d')    
        
    #print(df['date'])
    
    df['hr']    = pd.to_timedelta(df.iloc[:,3], unit = 'h')
    df['min']   = pd.to_timedelta(df.iloc[:,4], unit = 'm')
    
    df['Date']  = df['date'] + df['hr'] + df['min']
    df = df.sort_values(by="Date")
    df = df.set_index(df['Date'])
    
    df = df.drop(columns=['DD', 'MM', 'YYYY', 'hh', 'mm', 'date', \
                          'hr', 'min', 'dt_tmp', 'Date'])
    df = df.reindex(idx)
    data = df.iloc[:,1]

    return data
