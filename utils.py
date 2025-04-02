# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:39:56 2025

@author: bugatap
"""


import numpy as np

from helper_code import load_signals

from scipy.signal import resample, resample_poly

import mne

# helper function to determine file path
def get_path(base_path, source, part):
    subfolder = ''
    if source == 'CODE-15':
        subfolder =  'CODE-15/wfdb_chagas_full'
    if source == 'SamiTrop':
        subfolder =  'SamiTrop/wfdb'
    assert subfolder != ''
    path = base_path + '/' + subfolder
    if part > -1:
        path += '/part{}'.format(part)
    return path

# filtracia 
def filter_signal(X, fs):
    # notch filter
    # mne requires signal ion shape (n_channels, n_ts)
    # filtracia frekvencie striedaveho prudu (v Brazilii 60 Hz)
    X = X.T
    X = mne.filter.notch_filter(X, fs, 50, n_jobs=1, verbose='error')
    X = mne.filter.notch_filter(X, fs, 60, n_jobs=1, verbose='error')
        
    # bandpass filter
    X = mne.filter.filter_data(X, fs, 0.1, 30.0, n_jobs=1, verbose='error')    
    # transposing back to original shape
    return X.T 

# standardziacia signalu
def standardize_signal(signal):
    m = signal.mean(axis=0)
    s = signal.std(axis=0)
    s[s == 0] = 1
    signal = (signal - m) / s
    #print(m, s)
    return signal

# zmena frekvencie
def change_frequency(sig, fs, new_fs, poly=False):
    if fs == new_fs:
        return sig
    n_orig = len(sig)
    n_new = int(n_orig * new_fs/fs)
    if poly:
        sig = resample_poly(x=sig, up=2, down=2*fs/new_fs, axis=0)
    else:
        sig = resample(x=sig, num=n_new, axis=0)        
    
    return sig

# nacitanie jedneho signalu
def read_signal(base_path, source, part, ecg_id, standardize, target_fs):
    # determine path
    path = get_path(base_path, source, part)

    # load signal
    signal, fields = load_signals(path + '/' + str(ecg_id))
    #print(signal.shape)
    #print(fields)
    
    # odfiltrovanie frekvencie elektrickej siete a sumu
    fs = fields['fs']
    signal = filter_signal(signal, fs)
    
    # resample
    if fs != target_fs:
        signal = change_frequency(signal, fs, target_fs, poly=False)
    
    # standardization
    if standardize:
        signal = standardize_signal(signal)

    return signal    

