import scipy.io
import numpy as np
from gwpy.timeseries import TimeSeries


# load file (might be changed for different times)
file = scipy.io.loadmat('MLdata_L1HAM5_1381528818_4000_matrix_v2.mat')
time = 1381528818
data = file['data_matrix']

# define channels here
channels = ['L1:ISI-GND_STS_ITMY_X_DQ','L1:ISI-GND_STS_ITMY_Y_DQ','L1:ISI-GND_STS_ITMY_Z_DQ','L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Y_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Z_IN_DQ','L1:ISI-HAM5_BLND_CPSRX_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRY_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRZ_IN1_DQ','L1:ISI-HAM5_BLND_GS13X_IN1_DQ']

dic = {}
for idx,c in enumerate(channels):
    print(idx, c)
    dic[c] = data[:,idx]

# skip first 100 seconds of data to be sure that everything is in a stable state
selc = {}
for c in channels:
    selc[c] = dic[c][100*128:] 

# transfer data to time series to resample and bandpass
for c in channels:
    ts = TimeSeries(selc[c], dt=1/128., t0=time+100)
    # resample to 4Hz
    ts = ts.resample(4.0)
    # bandpass to 0.1-0.3 Hz
    ts = ts.bandpass(0.1,0.3)

    selc[c] = np.array(ts.value)

# combine the individual columns into one numpy array
l = [np.transpose(selc[c]) for c in channels]
data_prep = np.column_stack(l)

# split data in train and test set
n_sec = 3000
train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

data_prep_ = np.transpose(data_prep)
train_data = data_prep_[:, :int(4*n_sec*train_frac)]
val_data = data_prep_[:, int(4*n_sec*train_frac):int(4*n_sec*(train_frac+val_frac))]
test_data = data_prep_[:, int(4*n_sec*(train_frac+val_frac)):int(4*n_sec)]

# normalize data for training purposes
mean = train_data.mean(axis=-1, keepdims=True)
std = train_data.std(axis=-1, keepdims=True)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

# save data in numpy format
np.save('data/train_{}'.format(time), train_data)
np.save('data/val_{}'.format(time), val_data)
np.save('data/test_{}'.format(time), test_data)
