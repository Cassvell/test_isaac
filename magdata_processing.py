import pandas as pd
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from scipy.stats import genpareto, kstest #anderson
#from datetime import datetime
from scipy.optimize import curve_fit 
# Ajuste de distribuciones
import sys
from lmoments3 import distr
import lmoments3 as lm
import kneed as kn
from numpy.linalg import LinAlgError
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import NearestNDInterpolator
from magnetic_datstruct import get_dataframe
from scipy.signal import medfilt
from aux_time_DF import index_gen, convert_date
# =============================================================================
def POLY_FIT(x, y, ndegree, measure_errors=None, corrm_old=None,yband=None):
 
    n = len(x)
    if n != len(y):
        raise ValueError('X and Y must have the same number of elements.')

    m = ndegree + 1  # Number of elements in the coefficient vector

    
    status = 0
    # Compute standard deviations based on measurement errors if provided
    sdev = 1.0
    if measure_errors is not None:
        sdev *= measure_errors
    sdev2 = sdev**2

    # Construct work arrays
    covar = np.zeros((m, m), dtype=float)  # Least square matrix, weighted matrix
    b = np.zeros(m, dtype=float)  # Will contain sum weights*y*x^j
    z = 1.0  # Polynomial term (guarantees double precision calculation)
    wy = np.array(y, dtype=float)
    if measure_errors is not None:
        wy /= sdev2

 
    covar[0, 0] = np.sum(1.0 / sdev2) if measure_errors is not None else n
    b[0] = np.sum(wy)

    for p in range(1, 2 * ndegree + 1):
        z *= x
        if p < m:
            b[p] = np.sum(wy * z)
        for j in range(max(0, p - ndegree), min(ndegree, p) + 1):
            covar[j, p - j] = np.sum(z / sdev2) if measure_errors is not None else np.sum(z)

    try:
        covar = np.linalg.inv(covar)
    except LinAlgError:
        status = 1
        result =  np.full(m, np.nan)
       
        print("Singular matrix detected.")
      
 
            

    result = np.dot(b, covar)

    yfit = result[ndegree]
    for k in range(ndegree-1, -1, -1):
        yfit = result[k] + yfit * x
  
    sigma = np.sqrt(np.abs(np.diag(covar)))
    
#    print("result",result)
    if measure_errors is not None:
        diff = (yfit - y) ** 2
        chisq = np.sum(diff / sdev2)
        var = np.sum(diff) / (n - m) if n > m else 0
    else:
        chisq = np.sum((yfit - y) ** 2)
        var = chisq / (n - m) if n > m else 0
        sigma *= np.sqrt(chisq / (n - m))

    yerror = np.sqrt(var)
    
    if yband is not None:
        yband = np.full(n, np.nan)

        for p in range(1, 2 * ndegree + 1):
            z *= x
            sum_val = 0.0
            for j in range(max(0, p - ndegree), min(ndegree, p) + 1):
                sum_val += covar[j, p - j]
            yband += sum_val * z

        yband *= var

        if np.min(yband) < 0 or not np.all(np.isfinite(yband)):
            status = 3
            # if not corrm_old:
            #     raise Exception('Undefined (NaN) error estimate encountered.')
            # else:
            #     yerror_old.fill(0)
            #     yfit_old.fill(0)
            #     yband_old.fill(0)
            result =  np.full(m, np.nan)
        else:
            yband = np.sqrt(yband)


    class objeto(object):
        def __init__(self):
            self.result = None
            self.chisq = None
            self.covar = None
            self.sigma = None
            self.var = None
            self.yfit = None
            self.yband = None
            self.yerror = None
            self.status = None
        
    payload = objeto()
    
    payload.result = result[::-1]
    payload.chisq = chisq
    payload.sigma = sigma
    payload.var = var
    payload.yfit = yfit
    payload.yband = yband 
    payload.yerror = yerror
    payload.status = status
    return payload
###############################################################################
#generación del índice temporal Date time para las series de tiempo
###############################################################################   

###############################################################################
'''
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

filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)

###############################################################################}
path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 

'''
###############################################################################
#DETERMINACIÓN DE LINEAS BASE
###############################################################################
###############################################################################
###############################################################################

###############################################################################
#función para calcular linea base mensual
###############################################################################   
def base_line(data, idx, idx_daily):    
    ndata = len(data)
    ndays = int(ndata/1440)
    night_data = ndays*4
    hourly_sample = int(ndata/60) 
###############################################################################
#cálculo de moda móvil
###############################################################################  
    def hourly_mode(data):
        mode_hourly = []
        for i in range(hourly_sample):
            mod =  mode(data[i*60:(i+1)*60-1])
            mode_hourly.append(mod)
        return mode_hourly
    
    hourly_mode = hourly_mode(data)

    daily_stacked = []      #moda 

    ac_mode = np.zeros(ndays * 24)  

###############################################################################

    
    for i in range(ndays):
        if i == 0:
            # For the first day, use the first day and the next day
            tw_mode = hourly_mode[i*24:(i+2)*24]
            ac_mode[i*24:(i+2)*24] += tw_mode
        elif i == ndays - 1:
            # For the last day, use the previous day and the last day
            tw_mode = hourly_mode[(i-2)*24:(i+1)*24]
            ac_mode[(i-2)*24:(i+1)*24] += tw_mode
        else:
            # For all other days, use the previous day, the current day, 
            # and the next day
            tw_mode = hourly_mode[(i-1)*24:(i+2)*24]
            ac_mode[(i-1)*24:(i+2)*24] += tw_mode

        sum_mode = np.nanmean(tw_mode)
    #    print(np.nanmean(tw_mode))
    
        daily_stacked.append(sum_mode)
############################################################################### 
#excluimos los días donde IQR = supere el umbral
#obtenido del procedimiento anterior    
###############################################################################  
###############################################################################
#Determinación de umbral para la ventana de tiempo
###############################################################################
    #Determinamos primero un arreglo de picos, basado en MAX(IQR)
    pickwindow = [3,4,6,8,12]
    original_daily_stacked = np.copy(daily_stacked)
    disturbed_days_sample = []
    undisturbed_days_sample = []
    line_styles = ['-', '--', '-.', ':']
    for i in range(len(pickwindow)):     
        picks = max_IQR(data, 60, pickwindow[i]) #los picos al ser de 6 h, implica una muestra 
                                     #de ndays X 6 picos              
       # print(picks)
        x, GPD, threshold = get_threshold(picks)
        second_derivative = np.gradient(np.gradient(GPD))
        if not all([val < 0 for val in second_derivative]):
        
            daily_picks = max_IQR(data, 60, 24)   
            
            n = ndays
            #print(len(daily_picks), len(idx_daily))
            list_days = get_qd_dd(daily_picks, idx_daily, 'I_iqr', n) 
            i_iqr = list_days['VarIndex']
            daily_stacked = np.copy(original_daily_stacked)
        # Iterate over the daily_stacked array and apply the threshold
            for j in range(len(daily_stacked)):
                if i_iqr[j] >= threshold:
                    i_iqr[j] = np.nan
                    daily_stacked[j] = i_iqr[j]
                    disturbed_days_sample.append(j)
                else:
                    daily_stacked[j] = daily_stacked[j]
                    undisturbed_days_sample.append(j)

            daily_stacked = np.array(daily_stacked)  
           # print(daily_stacked)
            undisturbed_days = daily_stacked[~np.isnan(daily_stacked)]
            trials = len(undisturbed_days)
            
            style = line_styles[i % len(line_styles)]
            #plt.hist(picks, density=True, bins=ndays*2, histtype='stepfilled',\
            #         alpha=0.6)
            #plt.plot(x, GPD, lw=2, label=f'wide window: {pickwindow[i]} hr')
            #plt.axvline(x = threshold, color = 'k', \
            #            linestyle=style, label = f'kn = {threshold:.2f}')
            #plt.legend()            
           # print(trials)
            if trials > 9:
                break
        else:
            break
    #plt.show()        
    print(f'number of undisturbed days: {len(undisturbed_days)}')
###############################################################################
###############################################################################
#FILL GAPS BETWEEN EMPTY DAILY VALUES
###############################################################################       
    daily_data = np.linspace(0, len(daily_stacked)-1, len(daily_stacked))
  #  min_data = np.linspace(0, len(data)-1, len(data))     
    def nan_helper(y):    
        return np.isnan(y), lambda z: z.nonzero()[0]   
    nans, x= nan_helper(daily_stacked)    
# Create cubic interpolation function based on valid (non-NaN) data points
    valid_data_indices = x(~nans)
    valid_data_values = daily_stacked[~nans]
    cubic_interp = interp1d(valid_data_indices, valid_data_values, kind='cubic', fill_value="extrapolate")
    
    # Interpolate to fill NaN values in the original array
    daily_stacked[nans] = cubic_interp(x(nans))             
###############################################################################
# Daily interpolation
############################################################################### 
    interpol = splrep(daily_data, daily_stacked,k=3,s=5)
    # Evaluar la interpolación en puntos específicos
    time_axis = np.linspace(min(daily_data), max(daily_data), ndata)
    baseline_curve = splev(time_axis, interpol)
    baseline_line = [np.median(daily_stacked)]*ndata
    
    return baseline_line, baseline_curve, undisturbed_days_sample

'''    
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12,6), dpi=200)
    #fig, ax = plt.subplots(2)
    fig.suptitle('Ajuste de distribuciones', fontsize=24, \
                 fontweight='bold')

    ax.set_title('Distribución Gausiana', fontsize=18)
    ax.hist(data_nonan, bins=hourly_sample - 1, \
        density=True, alpha=0.6, color='g', label='Data')
    ax.plot(x_interval_for_fit, gaussfit, color='red', \
        lw=2, label='Fitted Gaussian')
    ax.axvline(x = popt[1], color = 'k', label = 'center of GF')     
    ax.grid()
    ax.set_xlabel('H component distribution [bins = 1 h]')
    ax.set_ylabel('prob', fontweight='bold')
    ax.legend()
    
    ax2.set_title('Distribución General de Pareto', fontsize=18)
    ax2.hist(picks, density=True, bins=ndays*2, histtype='stepfilled',\
             alpha=0.4)
    ax2.plot(x, GPD, 'r-', lw=2, label='Fitted GPD')    
    ax2.grid()
    ax2.set_xlabel('IQR Variation picks [bins = 12 h]', fontweight='bold')
    ax2.set_ylabel('prob', fontweight='bold')
    ax2.legend()

    fig.savefig("/home/isaac/tools/test/test_isaac/distributions/dist_fit"+\
                str(year)+str(month)+'.png')
    plt.show()    
    return base_line    
'''
###############################################################################
#diurnal variation computation
###############################################################################
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
    #qdp = []
    qd_list = get_qd_dd(iqr_picks, idx_daily, 'qdl', n)
    qdl = [[0] * 1440 for _ in range(n)]
    #print(data['2024-02-04'])
    #baseline = [[0] * 240 for _ in range(n)]
    baseline = []
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
        baseline.append(np.nanmedian(qd_2h))
        #print(baseline)
        qdl[i] = qdl[i]-np.nanmedian(baseline)
        
        print(qd, ' | ',  max(qdl[i]))
      #  plt.plot(qdl[i], label=i+': '+qd) 
    
    # Convert qdl to a numpy array for easy manipulation
    # Generate the average array
    qd_baseline = np.nanmedian(qdl, axis=0)
    
    #plt.plot(qd_baseline, color='k', linewidth=4.0, label = '<QDL>')
    qd_hourly_sample = []

    for i in range(int(len(qd_baseline)/(60))):
    #    print(qd_baseline[i*60:(i+1)*60-1])
        mod = mode(qd_baseline[i*60:(i+1)*60-1])
        qd_hourly_sample.append(mod)
    x = np.linspace(0,23,24)
    
    diurnal_baseline = np.tile(qd_hourly_sample, ndays)        

    #agregar un proceso extra para eliminar artefacto    
    interpol = splrep(tw, diurnal_baseline,k=3,s=5)
    
    # Evaluar la interpolación en puntos específicos
    time_axis = np.linspace(min(tw), max(tw), ndata)
    QD_baseline_min = splev(time_axis, interpol)
    
    mw = 61
    median_filtered = medfilt(QD_baseline_min, kernel_size=mw)
    kernel = np.ones(mw) / mw
    diurnal_variation = np.convolve(median_filtered, kernel, mode='same')
    
#    print(diurnal_baseline_min)

    print(f'max amplitud value for <QDL>: {max(qd_baseline)}')
    print(f'max amplitud value for fit: {max(QD_baseline_min)}')
###############################################################################
 
    #path = '/home/isaac/tools/test/test_isaac/qdl/'
    #plt.title('St: '+st+', Period: '+month, fontweight='bold', fontsize=18)
    #plt.plot(QD_baseline_min[0:1439], color='b', linewidth=4.0, label = 'Ajuste')
    #plt.xlim(0,1440)
    #plt.xlabel('Time [UTC]', fontweight='bold', fontsize=16)
    #plt.ylabel('H [nT]', fontweight='bold', fontsize=16)    
    #plt.legend()
    #plt.savefig(path+month+'_'+st, dpi=200)
    qd_offset = np.nanmedian(baseline)
    return diurnal_variation, qd_offset
###############################################################################
'''
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

    knp = None
    for i in range(len(qd_hourly_sample)):
        if qd_hourly_sample[i] >= knee_point:
            knp = i
            break
            #print(QD_sample1)
    QD_sample1 = qd_hourly_sample[0:knp]
    QD_sample2 = qd_hourly_sample[knp:24]
    

    x1 = np.linspace(0,len(QD_sample1)-1, len(QD_sample1))
    QDH1 = POLY_FIT(x1, QD_sample1, ndegree=5)
    fit1 = QDH1.yfit    
    plt.plot(x1, QD_sample1, 'o')
    plt.plot(fit1)
    plt.show()
    
    x2 = np.linspace(0,len(QD_sample2)-1, len(QD_sample2))
    QDH2 = POLY_FIT(x2, QD_sample2, ndegree=5)
    fit2 = QDH2.yfit
    plt.plot(x2, QD_sample2, 'o')
    plt.plot(fit2)    
    plt.show()    

    QD = [x for n in (fit1,fit2) for x in n]    

    #plt.plot(QD)
    QD_baseline = np.tile(QD, ndays)  
    
    #agregar un proceso extra para eliminar artefacto    
    interpol = splrep(tw, QD_baseline,k=3,s=5)
    
    # Evaluar la interpolación en puntos específicos
    time_axis = np.linspace(min(tw), max(tw), ndata)
    QD_baseline_min = splev(time_axis, interpol)
   # month = str(qd_list[0])[0:7]

    plt.plot(QD)
    plt.show()
'''
###############################################################################
#FUNCIONES AUXILIARES
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#   COMPUTATION OF THE THRESHOLD
###############################################################################
def get_threshold(picks):
    ndays = int(len(picks)/4)

    picks = np.array(picks)  
    picks = picks[~np.isnan(picks)]
    
    hist, bins = np.histogram(picks, bins=ndays*2, \
                                    density=True)  
 
   # print(np.isnan(picks).sum())  # Check for NaNs
  #  print(np.isinf(picks).sum())  # Check for Infs
 #   L_moments = lm.lmom_ratios(picks, nmom=3)
    GPD_paramet = distr.gpa.lmom_fit(picks)
    shape = GPD_paramet['c']
    threshold = GPD_paramet['loc']
    scale = GPD_paramet['scale']
    
    x = np.linspace(min(picks), max(picks), len(picks))    
    GPD =  genpareto.pdf(x, shape, loc=threshold, scale=scale)
    GPD = np.array(GPD)
    if any(v == 0.0 for v in GPD):
        GPD =  genpareto.pdf(x, shape, loc=min(bins), scale=scale)
   # print(GPD)
   
    params = genpareto.fit(picks)
    D, p_value = kstest(picks, 'genpareto', args=params)
    print(f"K-S test result:\nD statistic: {D}\np-value: {p_value}")
    
# Interpretation of the p-value y TEST KS para evaluar distribución de picos
#IQR
    alpha = 0.05
    if p_value > alpha:
        print("Fail to reject the null hypothesis: data follows the GPD")
    else:
        print("Reject the null hypothesis: data does not follow the GPD")   
        
    kneedle = kn.KneeLocator(
        x,
        GPD,
        curve='convex',
        direction='decreasing',
        S=5,
        online=True,
        interp_method='interp1d',
    )
    knee_point = kneedle.knee #elbow_point = kneedle.elbow
    print(f'knee point: {knee_point}')
#El threshold será considerado como el punto de rodilla del ajuste de GPD.    
    return x, GPD, knee_point
###############################################################################
#ANDERSON-DARLING test for the right tail
###############################################################################
###############################################################################
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
def get_qd_dd(data, idx_daily, type_list, n):
    daily_var = {'Date': idx_daily, 'VarIndex': data}
    
    local_var = pd.DataFrame(data=daily_var)
    local_var = local_var.sort_values(by = "VarIndex", ignore_index=True)
    
    if type_list == 'qdl':
        local_var = local_var[0:n]['Date']   
    elif type_list == 'I_iqr':
        local_var = local_var.sort_values(by = "Date", ignore_index=True)
    return local_var
###############################################################################
#We call the base line derivation procedures
############################################################################### 
'''
data = get_dataframe(filenames, path, idx, dates)

#monthly base line
baseline_line, baseline_curve = base_line(data, idx, idx_daily) 

H_detrend1 = data-baseline_line
H_detrend2 = data-baseline_curve
#plt.plot(H_detrend)
#plt.show()

#diurnal base line
diurnal_baseline, offset = get_diurnalvar(H_detrend1, idx_daily, st)
#H component, 1 min. resolution
H = H_detrend1-diurnal_baseline
#plt.plot(H)
H_noff1 = H-offset

dst = []
hr = int(len(data)/60)
for i in range(hr):
    tmp_h = np.nanmedian(H_noff1[i*60:(i+1)*60])
    dst.append(tmp_h)
    

fig, ax = plt.subplots(5, figsize=(12,8), dpi = 300) 
fig.suptitle(st+' Geomagnetic ST' , fontsize=24, \
             fontweight='bold') 
inicio = data.index[0]
final =  data.index[-1]
ax[0].plot(data.index, data, label='raw data')
ax[0].plot(data.index, baseline_curve, color='r', label='base offset tendency')
ax[0].axhline(y = baseline_line[0], color='g', label='base line monthly tendency')
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_ylabel('BH [nT]', fontweight='bold')
ax[0].legend()

ax[1].plot(data.index, H_detrend1, label='H - base line')
ax[1].plot(data.index, diurnal_baseline, color='r', label='diurnal variation')
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_ylabel('BH [nT]', fontweight='bold')
ax[1].legend()

ax[2].plot(data.index, H_detrend2, label='H - base curve')
ax[2].plot(data.index, diurnal_baseline, color='r', label='diurnal variation')
ax[2].grid()
ax[2].set_xlim(inicio,final)
ax[2].set_ylabel('BH [nT]', fontweight='bold')
ax[2].legend()

ax[3].plot(data.index, H_noff1, color='k', \
           label='H - (diurnal baseline + baseline+offset)')
ax[3].grid()
ax[3].set_xlim(inicio,final)
ax[3].set_ylabel(' BH [nT]', fontweight='bold')
ax[3].legend()

ax[4].plot(idx_hr, dst, color='k', label='Dst')
ax[4].grid()
ax[4].set_xlim(inicio,final)
ax[4].set_ylabel(' BH [nT]', fontweight='bold')
ax[4].legend()
fig.savefig("/home/isaac/tools/test/test_isaac/processed/"+\
            st+'_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.tight_layout() 
plt.show()
#dat = {'H raw' : data, 'baseline line' : baseline_line, \
#       'baseline cubeint' : baseline_curve, 'H det' : H_detrend1, 'H' : H, \
#        'dst' : dst   }
    
#df = pd.DataFrame(dat)        
#df.to_csv("/home/isaac/tools/test/test_isaac/processed/enero.csv")
'''