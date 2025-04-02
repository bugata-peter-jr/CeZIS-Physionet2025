#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os
import sys

from helper_code import *

import torch
from resnet_config import Config
from network import Network
from utils import filter_signal, change_frequency, standardize_signal

import pandas as pd

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    return

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    eval_mode = False
    model = Model(eval_mode=eval_mode)
    model.load(model_folder)
    if verbose:
        print('Model created and loaded.', flush=True)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    return model.predict(record)

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

class Model(object):
    def __init__(self, eval_mode=False):
        # configuration
        self.cfg = Config()
        
        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # networks
        self.networks = []
        self.networks_d = []

        # eval mode
        self.eval_mode = eval_mode
        if self.eval_mode:
            folds_df = pd.read_csv('./folds.csv')
            folds_df.exam_id = folds_df.exam_id.astype(str)
            self.fold_dict = {row['exam_id']:row['fold'] for _, row in folds_df.iterrows()}

    # load model from folder    
    def load(self, folder):
        cfg = self.cfg
        
        pretrained_folder = './pretrained'
                
        # create and load networks - classifier
        for i in range(5):

            # make resnet model
            network = Network(input_channels=cfg.leads, kernel=cfg.kernel, width=cfg.width, 
                              normalization=cfg.normalization, dropout=cfg.dropout, n_groups=cfg.n_groups)

            # to device
            network = network.to(device=self.device)
           
            # to evaluation mode
            network.eval()
            
            fname = pretrained_folder + '/classifier/weights{:d}.h5'.format(i)
            try:
                state_dict = torch.load(fname, map_location=self.device)
                network.load_state_dict(state_dict, strict=True)
            except:
                print('Weights file: {:s} not available.'.format(fname))
                exit(-1)
            
            self.networks.append(network) 

        # create and load networks - discriminator
        for i in range(5):
            # make resnet model
            network_d = Network(input_channels=cfg.leads, kernel=cfg.kernel, width=cfg.width, 
                              normalization=cfg.normalization, dropout=cfg.dropout, n_groups=cfg.n_groups)
            
            # to device
            network_d = network_d.to(device=self.device)
           
            # to evaluation mode
            network_d.eval()
            
            fname = pretrained_folder + '/discriminator/weights{:d}.h5'.format(i)
            try:
                state_dict = torch.load(fname, map_location=self.device)
                network_d.load_state_dict(state_dict, strict=True)
            except:
                print('Weights file: {:s} not available.'.format(fname))
                exit(-1)
            
            self.networks_d.append(network_d) 


    # predict
    def predict(self, record):
        cfg = self.cfg
        
        #print('record:', record)
        
        # load signal
        orig_signal, fields = load_signals(record)
        #print('orig signal.shape:', orig_signal.shape)
        
        # filter utility frequency
        fs = fields['fs']
        preproc_signal = filter_signal(orig_signal, fs)
        
        # resample
        if fs != cfg.fs:
            #print(cfg.fs)
            preproc_signal = change_frequency(preproc_signal, fs, cfg.fs, poly=False)
        
        # standardization
        if cfg.standardize_sig:
            preproc_signal = standardize_signal(preproc_signal)
            
        inplen = int(cfg.inplen * cfg.fs)
        inplen_d = int(6.4 * cfg.fs)
        siglen, n_leads = preproc_signal.shape
        #print('preproc signal.shape:', preproc_signal.shape)
        
        n_repeats = cfg.test_n_reps
        signals = []
        signals_d = []
        # n_repeats slices
        for j in range(n_repeats):
            # deterministic offset
            if siglen > inplen:
                if n_repeats > 1:
                    offset = int((siglen-inplen)/(n_repeats - 1)) * j
                else:
                    offset = int((siglen-inplen)/2)     
                signal = preproc_signal[offset:offset+inplen,:]            
            # padding
            else:
                k = inplen - siglen
                if n_repeats > 1:
                    k1 = int(k/(n_repeats - 1)) * j
                else:
                    k1 = int(k/2) 
                k2 = k - k1
                signal = np.concatenate([np.zeros(shape=(k1, n_leads)), preproc_signal, np.zeros(shape=(k2, n_leads))]) 
                
            signal = signal.T
            
            signal_t = torch.as_tensor(signal, dtype=torch.float32, device=self.device)
            signals.append(signal_t)

            # deterministic offset
            if siglen > inplen_d:
                if n_repeats > 1:
                    offset = int((siglen-inplen_d)/(n_repeats - 1)) * j
                else:
                    offset = int((siglen-inplen_d)/2)     
                signal_d = preproc_signal[offset:offset+inplen_d,:]            
            # padding
            else:
                k = inplen_d - siglen
                if n_repeats > 1:
                    k1 = int(k/(n_repeats - 1)) * j
                else:
                    k1 = int(k/2) 
                k2 = k - k1
                signal_d = np.concatenate([np.zeros(shape=(k1, n_leads)), preproc_signal, np.zeros(shape=(k2, n_leads))]) 
                
            signal_d = signal_d.T
            
            signal_td = torch.as_tensor(signal_d, dtype=torch.float32, device=self.device)
            signals_d.append(signal_td)
                        
        # input for network - classifier
        X = torch.stack(signals, dim=0)
        #print(X.shape)

        # input for network - discriminator
        X_d = torch.stack(signals_d, dim=0)
        #X_d = X[:,:,:inplen_d]
        #print(X_d.shape)
        
        # get fold for eval. mode
        if self.eval_mode:
            if record.count('/') > 0:
                idx = record.rindex('/') + 1
                ecg_id = record[idx:]
            else:
                ecg_id = record
            fold = self.fold_dict.get(ecg_id, -1)
            #print('ecg_id:', ecg_id, 'fold:', fold)
        else:
            fold = -1
        
        # prediction of all networks - classifier
        # or one network in eval. mode
        with torch.no_grad():
            probs = []       
            for i in range(len(self.networks)):
                if self.eval_mode and fold != -1 and i != fold:
                    continue
                network = self.networks[i]        
                output = network(X)
                output_s = torch.sigmoid(output)
                prob_for_net = torch.mean(output_s) 
                probs.append(prob_for_net.item())
            
        # get final prob and binary output
        final_prob = sum(probs) / len(probs)
        #print('probs:', probs)
        #print('final prob:', final_prob)

        # prediction of all networks - discriminator 
        # or one network in eval. mode
        with torch.no_grad():
            probs_d = []       
            for i in range(len(self.networks_d)):
                if self.eval_mode and fold != -1 and i != fold:
                    continue
                network_d = self.networks_d[i]        
                output_d = network_d(X_d)
                output_sd = torch.sigmoid(output_d)
                prob_for_net_d = torch.mean(output_sd) 
                probs_d.append(prob_for_net_d.item())
        brazil_prob = sum(probs_d) / len(probs_d)
        #print('probs_d:', probs_d)
        #print('brazil prob:', brazil_prob)
        
        if brazil_prob < 0.5:
            final_prob *= 0.001
        #print('final prob (corrected):', final_prob)

        binary_output = final_prob >= 0.5
        return binary_output, final_prob
            

# Save your trained model.
def save_model(model_folder, model):
    return