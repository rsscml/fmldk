#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import itertools
import random
from datetime import datetime, timedelta
import os
from pathlib import Path
import csv
import re
import gc
import math as m
import time
# visualization imports
from bokeh.plotting import figure, output_file, show, output_notebook, save
from bokeh.models import ColumnDataSource, HoverTool, Div, FactorRange
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Category10, Category20, Colorblind
from bokeh.io import reset_output
from bokeh.models.ranges import DataRange1d

class tft_dataset:
    def __init__(self, 
                 col_dict, 
                 window_len, 
                 fh, 
                 batch, 
                 min_nz,
                 scaling_method = 'mean_scaling',
                 interleave=1, 
                 PARALLEL_DATA_JOBS=1, 
                 PARALLEL_DATA_JOBS_BATCHSIZE=64, 
                 WORKDIR='/tmp/'):
        """
        col_dict: dictionary of various column groups {id_col:'',
                                                       target_col:'',
                                                       time_index_col:'',
                                                       datetime_format:'',
                                                       static_num_col_list:[],
                                                       static_cat_col_list:[],
                                                       temporal_known_num_col_list:[],
                                                       temporal_unknown_num_col_list:[],
                                                       temporal_known_cat_col_list:[],
                                                       temporal_unknown_cat_col_list:[],
                                                       strata_col_list:[],
                                                       sort_col_list:[],
                                                       wt_col:''}
        window_len: history_len + forecast_horizon
        fh: forecast_horizon
        batch: batch_size (per strata)
        min_nz: min. no. of non-zeros in the target input series to be eligible for train/test batch
        
        """
        self.col_dict = col_dict
        self.window_len = window_len
        self.fh = fh
        self.batch = batch
        self.min_nz = min_nz
        self.interleave = interleave
        self.scaling_method = scaling_method
        self.PARALLEL_DATA_JOBS = PARALLEL_DATA_JOBS
        self.PARALLEL_DATA_JOBS_BATCHSIZE = PARALLEL_DATA_JOBS_BATCHSIZE
        self.workdir = str(WORKDIR).rstrip('/\\').strip()
       
        # extract columnsets from col_dict
        self.id_col = self.col_dict.get('id_col', None)
        self.target_col = self.col_dict.get('target_col', None)
        self.time_index_col = self.col_dict.get('time_index_col', None)
        self.datetime_format = self.col_dict.get('datetime_format', None)
        self.strata_col_list = self.col_dict.get('strata_col_list', [])
        self.sort_col_list = self.col_dict.get('sort_col_list', [])
        self.wt_col = self.col_dict.get('wt_col', None)
        self.static_num_col_list = self.col_dict.get('static_num_col_list', [])
        self.static_cat_col_list = self.col_dict.get('static_cat_col_list', [])
        self.temporal_known_num_col_list = self.col_dict.get('temporal_known_num_col_list', [])
        self.temporal_unknown_num_col_list = self.col_dict.get('temporal_unknown_num_col_list', [])
        self.temporal_known_cat_col_list = self.col_dict.get('temporal_known_cat_col_list', [])
        self.temporal_unknown_cat_col_list = self.col_dict.get('temporal_unknown_cat_col_list', [])

        if (self.id_col is None) or (self.target_col is None) or (self.time_index_col is None):
            raise ValueError("Id Column, Target Column or Index Column not specified!")

        # full columnset for train/test/infer
        self.col_list = [self.id_col] + [self.target_col] + \
                        self.static_num_col_list + self.static_cat_col_list + \
                        self.temporal_known_num_col_list + self.temporal_unknown_num_col_list + \
                        self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        self.cat_col_list = self.static_cat_col_list + self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list

        # get columnset indices
        self.id_index =  self.col_list.index(self.id_col)
        self.target_index = self.col_list.index(self.target_col)
        self.static_num_indices = [i for i in range(len(self.col_list)) if self.col_list[i] in self.static_num_col_list]
        self.static_cat_indices = [i for i in range(len(self.col_list)) if self.col_list[i] in self.static_cat_col_list]
        self.temporal_known_num_indices = [i for i in range(len(self.col_list)) if self.col_list[i] in self.temporal_known_num_col_list]
        self.temporal_unknown_num_indices = [i for i in range(len(self.col_list)) if self.col_list[i] in self.temporal_unknown_num_col_list]
        self.temporal_known_cat_indices = [i for i in range(len(self.col_list)) if self.col_list[i] in self.temporal_known_cat_col_list]
        self.temporal_unknown_cat_indices = [i for i in range(len(self.col_list)) if self.col_list[i] in self.temporal_unknown_cat_col_list]
        
        # column indices dict
        self.col_index_dict = {'id_index': [self.id_col, self.id_index],
                               'target_index': [self.target_col, self.target_index],
                               'static_num_indices': [self.static_num_col_list, self.static_num_indices],
                               'static_cat_indices': [self.static_cat_col_list, self.static_cat_indices],
                               'temporal_known_num_indices': [self.temporal_known_num_col_list, self.temporal_known_num_indices],
                               'temporal_unknown_num_indices': [self.temporal_unknown_num_col_list, self.temporal_unknown_num_indices],
                               'temporal_known_cat_indices': [self.temporal_known_cat_col_list, self.temporal_known_cat_indices],
                               'temporal_unknown_cat_indices': [self.temporal_unknown_cat_col_list, self.temporal_unknown_cat_indices]} 
        
        
    def get_keys(self, data):
        """
        if strata_col specified:
            Returns a dictionary with strata_col values as Keys & corresponding id_col array as Values
        else:
            Returns a dictionary with only one item, a 'null' Key & an array of id_col values
        """
        def wt_fn(x):
            return x.groupby([self.id_col], sort=True)[self.wt_col].first().values
    
        if len(self.strata_col_list)>0:
            keys_dict = data.groupby(self.strata_col_list, sort=True)[self.id_col].unique().to_dict()
            if self.wt_col:
                wts_dict = data.groupby(self.strata_col_list, sort=True).apply(lambda x: wt_fn(x)).to_dict()
            else:
                wts_dict = data.groupby(self.strata_col_list, sort=True)[self.id_col].apply(lambda x: np.ones(x.nunique()) ).to_dict()
        else:
            keys_dict = {'ID': sorted(data[self.id_col].unique())}
            if self.wt_col:
                wts_dict = {'ID': data.groupby([self.id_col], sort=True)[self.wt_col].mean().values}
            else:
                wts_dict = {'ID': np.ones(data[self.id_col].nunique())}
                
        return keys_dict, wts_dict

    def dict_sampler(self, v, w):
        """
        random sample of size 'batch' from an array returned as list
        """
        w = w/np.sum(w) # ensure sum of probabilities==1

        if self.batch > len(v):
            replace = True # if not enough items (less than batch size, sample with replace) 
        else:
            replace = False
            
        return list(np.random.choice(v, size=self.batch, replace=replace, p=w))

    def select_ids(self, keys_dict, wts_dict):
        """
        Sample Ids for a train/test iteration from the Keys_dict
        """
        n_jobs = min(len(keys_dict),4)
        sample_id = Parallel(n_jobs=n_jobs, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(self.dict_sampler)(v1,v2) for (k1,v1),(k2,v2) in zip(keys_dict.items(), wts_dict.items()) )
        sample_id = list(itertools.chain.from_iterable(sample_id))
        random.shuffle(sample_id)
        return sample_id

    def select_arrs(self, df, mode):
        """
        Extract train/test samples from each sampled Id as 2-D arrays [window_len, num_columns]
        """
        # filter out ids with insufficient timestamps (at least one datapoint should be before train cutoff period)
        df = df.groupby(self.id_col).filter(lambda x: x[self.time_index_col].min()<self.train_till)

        groups = df.groupby([self.id_col])
        sampled_arr = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(self.df_sampler)(gdf, mode) for _,gdf in groups)
        arr_list = [tup[0] for tup in sampled_arr]
        pad_list = [tup[1] for tup in sampled_arr]
        scale_list = [tup[2] for tup in sampled_arr]
        known_scale_list = [tup[3] for tup in sampled_arr]
        unknown_scale_list = [tup[4] for tup in sampled_arr]
        arr = np.stack(arr_list, axis=0)
        pad_arr = np.stack(pad_list, axis=0)
        scale_arr = np.stack(scale_list, axis=0)
        known_scale_arr = np.stack(known_scale_list, axis=0)
        unknown_scale_arr = np.stack(unknown_scale_list, axis=0)
        return arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr

    def df_sampler(self, gdf, mode):
        """
        Helper function for select_arrs
        
        gdf = gdf.reset_index(drop=True)
        first_nonzero_index = gdf[self.target_col].ne(0).idxmax()
        gdf = gdf.iloc[first_nonzero_index:,:]
        gdf = gdf.reset_index(drop=True)
        
        """
        
        scale_gdf = gdf[gdf[self.time_index_col]<=self.train_till].reset_index(drop=True)
        
        if self.scaling_method == 'mean_scaling':
            target_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.target_col])), 1.0)
            target_sum = np.sum(np.abs(scale_gdf[self.target_col]))
            scale = [np.divide(target_sum, target_nz_count) + 1.0]
            
            if len(self.temporal_known_num_indices) > 0:
              known_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
              known_sum = np.sum(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0)
              known_scale = [np.divide(known_sum, known_nz_count) + 1.0]
            else:
              known_scale = [1]

            if len(self.temporal_unknown_num_indices) > 0:
              unknown_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0), 1.0)
              unknown_sum = np.sum(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0)
              unknown_scale = [np.divide(unknown_sum, unknown_nz_count) + 1.0]
            else:
              unknown_scale = [1]

        elif self.scaling_method == 'standard_scaling':
            scale_mu = scale_gdf[self.target_col].mean()
            scale_std = np.maximum(scale_gdf[self.target_col].std(), 0.0001)
            scale = [scale_mu, scale_std]

            if len(self.temporal_known_num_indices) > 0:
              known_mean = np.mean(scale_gdf[self.temporal_known_num_col_list].values, axis=0)
              known_stddev = np.maximum(np.std(scale_gdf[self.temporal_known_num_col_list].values, axis=0), 0.0001)
              known_scale = [known_mean, known_stddev]
            else:
              known_scale = [0, 1]

            if len(self.temporal_unknown_num_indices) > 0:
              unknown_mean = np.mean(scale_gdf[self.temporal_unknown_num_col_list].values, axis=0)
              unknown_stddev = np.maximum(np.std(scale_gdf[self.temporal_unknown_num_col_list].values, axis=0), 0.0001)
              unknown_scale = [unknown_mean, unknown_stddev]
            else:
              unknown_scale = [0, 1]

        elif self.scaling_method == 'no_scaling' or self.scaling_method == 'log_scaling':
            scale = [1.0]
            known_scale = [1.0]
            unknown_scale = [1.0]
        
        if mode == 'train':
            gdf = gdf[gdf[self.time_index_col]<=self.train_till].reset_index(drop=True)
        elif mode == 'test':
            test_len = int(gdf[(gdf[self.time_index_col]>self.train_till) & (gdf[self.time_index_col]<=self.test_till)].groupby(self.id_col)[self.target_col].count().max())
            test_len = test_len + (self.window_len - self.fh) 
            gdf = gdf[gdf[self.time_index_col]<=self.test_till].groupby(self.id_col).apply(lambda x: x[-test_len:]).reset_index(drop=True)
            
        gdf = gdf.reset_index(drop=True)
        delta = len(gdf) - self.window_len
        
        if delta<0:
            arr = gdf.loc[:,self.col_list].reset_index(drop=True).values
            # left pad to window_len with 0
            pad_len = np.abs(delta)
            arr = np.pad(arr,pad_width=((pad_len,0),(0,0)), mode='constant', constant_values=0)
        elif delta==0:
            pad_len = 0
            rand_start = 0
            arr = gdf.loc[rand_start:rand_start + self.window_len - 1, self.col_list].reset_index(drop=True).values
        else:
            pad_len = 0
            rand_start = random.randrange(delta)
            arr = gdf.loc[rand_start:rand_start + self.window_len - 1, self.col_list].reset_index(drop=True).values
            
        return (arr, pad_len, scale, known_scale, unknown_scale)
    
    def df_infer_sampler(self, gdf):
        """
        Helper function for select_all_arrs
        """
        scale_gdf = gdf[gdf[self.time_index_col]<=self.train_till].reset_index(drop=True)
        
        if self.scaling_method == 'mean_scaling':
            target_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.target_col])), 1.0)
            target_sum = np.sum(np.abs(scale_gdf[self.target_col]))
            scale = [np.divide(target_sum, target_nz_count) + 1.0]

            if len(self.temporal_known_num_indices) > 0:
              known_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0), 1.0)
              known_sum = np.sum(np.abs(scale_gdf[self.temporal_known_num_col_list].values), axis=0)
              known_scale = [np.divide(known_sum, known_nz_count) + 1.0]
            else:
              known_scale = [1]

            if len(self.temporal_unknown_num_indices) > 0:
              unknown_nz_count = np.maximum(np.count_nonzero(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0), 1.0)
              unknown_sum = np.sum(np.abs(scale_gdf[self.temporal_unknown_num_col_list].values), axis=0)
              unknown_scale = [np.divide(unknown_sum, unknown_nz_count) + 1.0]
            else:
              unknown_scale = [1]

        elif self.scaling_method == 'standard_scaling':
            scale_mu = scale_gdf[self.target_col].mean()
            scale_std = np.maximum(scale_gdf[self.target_col].std(), 0.0001)
            scale = [scale_mu, scale_std]

            if len(self.temporal_known_num_indices) > 0:
              known_mean = np.mean(scale_gdf[self.temporal_known_num_col_list].values, axis=0)
              known_stddev = np.maximum(np.std(scale_gdf[self.temporal_known_num_col_list].values, axis=0), 0.0001)
              known_scale = [known_mean, known_stddev]
            else:
              known_scale = [0, 1]

            if len(self.temporal_unknown_num_indices) > 0:
              unknown_mean = np.mean(scale_gdf[self.temporal_unknown_num_col_list].values, axis=0)
              unknown_stddev = np.maximum(np.std(scale_gdf[self.temporal_unknown_num_col_list].values, axis=0), 0.0001)
              unknown_scale = [unknown_mean, unknown_stddev]
            else:
              unknown_scale = [0, 1]

        elif self.scaling_method == 'no_scaling' or self.scaling_method == 'log_scaling':
            scale = [1.0]
            known_scale = [1.0]
            unknown_scale = [1.0]
            
        # restrict to prediction window_len
        gdf = gdf[gdf[self.time_index_col]<=self.future_till].groupby(self.id_col).apply(lambda x: x[-self.window_len:]).reset_index(drop=True)
        gdf = gdf.reset_index(drop=True)
        
        delta = len(gdf) - self.window_len
        if delta<0:
            arr = gdf.loc[:,self.col_list].reset_index(drop=True).values
            pad_len = np.abs(delta)
            arr = np.pad(arr,pad_width=((pad_len,0),(0,0)), mode='constant', constant_values=0)
            date_arr = gdf.loc[:,self.time_index_col].reset_index(drop=True).values
            date_arr = np.pad(date_arr,pad_width=((pad_len,0)), mode='constant', constant_values=0)
            date_arr = date_arr[-self.fh:]
        elif delta==0:
            pad_len = 0
            rand_start = 0
            arr = gdf.loc[rand_start:rand_start + self.window_len - 1, self.col_list].reset_index(drop=True).values
            date_arr = gdf.loc[rand_start:rand_start + self.window_len - 1, self.time_index_col].reset_index(drop=True).tail(self.fh).values
        else:
            pad_len = 0
            rand_start = random.randrange(delta)
            arr = gdf.loc[rand_start:rand_start + self.window_len - 1, self.col_list].reset_index(drop=True).values
            date_arr = gdf.loc[rand_start:rand_start + self.window_len - 1, self.time_index_col].reset_index(drop=True).tail(self.fh).values
        return (arr, pad_len, date_arr, scale, known_scale, unknown_scale)
    
    def select_all_arrs(self, data):
        """
        Use for inference dataset
        """
        # filter out ids with insufficient timestamps (at least one datapoint should be before history cutoff period)
        data = data.groupby(self.id_col).filter(lambda x: x[self.time_index_col].min()<self.history_till)

        groups = data.groupby([self.id_col])
        sampled_arr = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(self.df_infer_sampler)(gdf) for _,gdf in groups)
        arr_list = [tup[0] for tup in sampled_arr]
        pad_list = [tup[1] for tup in sampled_arr]
        date_list = [tup[2] for tup in sampled_arr]
        scale_list = [tup[3] for tup in sampled_arr]
        known_scale_list = [tup[4] for tup in sampled_arr]
        unknown_scale_list = [tup[5] for tup in sampled_arr]

        arr = np.stack(arr_list, axis=0)
        pad_arr = np.stack(pad_list, axis=0)
        date_arr = np.stack(date_list, axis=0)
        scale_arr = np.stack(scale_list, axis=0)
        known_scale_arr = np.stack(known_scale_list, axis=0)
        unknown_scale_arr = np.stack(unknown_scale_list, axis=0)

        id_arr = arr[:,-1:,0]
        
        return arr, pad_arr, id_arr, date_arr, scale_arr, known_scale_arr, unknown_scale_arr
    
    def sort_dataset(self, data):
        """
        sort pandas dataframe by provided col list & order
        """
        if len(self.sort_col_list) > 0:
            data = data.sort_values(by=self.sort_col_list, ascending=True)
        else:
            pass
        return data
    
    def check_null(self, data):
        """
        Check for columns containing NaN
        """
        null_cols = []
        null_status = None
        for col in self.col_list:
            if data[col].isnull().any():
                null_cols.append(col)
        
        if len(null_cols)>0:
            null_status == True
        else:
            null_status == False
        
        return null_status, null_cols
    
    def data_generator(self, data, mode):
        """
        outputs:
        --------
        arr: [batch,window_len,len(col_list)]
        pad_arr as list: [pad_len_arr1, pad_len_arr2, ...]

        """
        
        train_data, test_data = self.split_train_test(data)
        
        keys_dict, wts_dict = self.get_keys(data)
        self.train_test_batch_size = int(self.batch*len(keys_dict))
       
        while True:
            sample_id = self.select_ids(keys_dict, wts_dict)
            df = data.query("{}==@sample_id".format(self.id_col))
            arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr = self.select_arrs(df, mode)
            
            #print("data gen: ", scale_arr.shape)
            # -- done for @tf.function retracing reason. May revisit later
            valid_indices = np.where(np.count_nonzero(arr[:,:self.window_len-self.fh,self.target_index].astype(np.float32)>0,axis=1)>=self.min_nz)
            arr = arr[valid_indices]
            pad_arr = pad_arr[valid_indices]
            scale_arr = scale_arr[valid_indices]
            known_scale_arr = known_scale_arr[valid_indices]
            unknown_scale_arr = unknown_scale_arr[valid_indices]

            model_in, model_out, scale, weights  = self.preprocess(arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr, mode)
            yield model_in.astype(str), model_out.astype(np.float32), scale.astype(np.float32), weights.astype(np.float32)
            
            
    def vocab_list(self, data):
        """
        Function to generate default embedding dimensions for categorical vars as 3rd root of no. of unique values.
        
        vocab_dict = {}
        embed_dict = {}
        
        for col in self.cat_col_list:
            col_unique_vals = data[col].astype(str).unique().tolist()
            vocab_dict[col] = col_unique_vals
            emb_dim = max(m.ceil(len(col_unique_vals)**0.33),2)
            embed_dict[col] = emb_dim
        """
        vocab_dict = {}
        for k,d in self.col_index_dict.items():
            if k in ['static_cat_indices','temporal_known_cat_indices','temporal_unknown_cat_indices']:
                if len(d[0])>0:
                    cols_val_list = []
                    cols_dim_list = []
                    for col in d[0]:
                        col_unique_vals = data[col].astype(str).unique().tolist()
                        emb_dim = max(m.ceil(len(col_unique_vals)**0.33),2)       
                        cols_val_list.append(col_unique_vals)
                        cols_dim_list.append(emb_dim)
                    vocab_dict[k] = [d[0], cols_val_list, cols_dim_list]
                           
        return vocab_dict
    
    def preprocess(self, sample_arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr, mode):
        """
        Preprocess_samples Tensorflow Ops & outputs resp.
        """
        sequence_len = int(self.window_len)
        max_input_len = int(self.window_len - self.fh)
        
        sid =  sample_arr[..., [self.id_index]].astype(str)
        target = sample_arr[..., [self.target_index]].astype(float)
        
        # target outlier correction
        #SU = np.maximum(3.0*np.quantile(target[:,:max_input_len,:], q=0.97, axis=1, keepdims=True), 0)
        #SL = np.minimum(np.quantile(target[:,:max_input_len,:], q=0.01, axis=1, keepdims=True), 0)
        #target = np.clip(target, a_min=SL, a_max=SU)
        
        # scale target : target/target_mean
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling'):
            target_nz_mean = scale_arr.reshape(-1,1,1)
            target_scaled = np.divide(target, target_nz_mean)
        elif self.scaling_method == 'standard_scaling':
            target_mean = scale_arr[:,0].reshape(-1,1,1)
            target_stddev = scale_arr[:,1].reshape(-1,1,1)
            target_scaled = np.divide(np.subtract(target, target_mean), target_stddev)
            target_scaled = np.nan_to_num(target_scaled) # correct where stddev is 0
        elif self.scaling_method == 'log_scaling':
            target_nz_mean = scale_arr.reshape(-1,1,1)
            target_scaled = np.log1p(target)
       
        # build model_in array  
        model_in = np.concatenate((sid, target_scaled), axis=-1)
        model_out = target_scaled[:,-self.fh:,:]
        
        if mode == 'infer':
            model_in[:,-self.fh:,1] = 0
        
        # make model_in target values zero for the fh
        #target_scaled[:,-self.fh:, :] = 0
        #model_in = np.concatenate((sid, target_scaled), axis=-1)
        
        if len(self.static_num_indices) > 0:
            static_num = sample_arr[..., self.static_num_indices].astype(float)
            model_in = np.concatenate((model_in, static_num), axis=-1)
            
        if len(self.static_cat_indices) > 0:
            static_cat = sample_arr[..., self.static_cat_indices].astype(str)
            model_in = np.concatenate((model_in, static_cat), axis=-1)

        if len(self.temporal_known_num_indices) > 0:
            known_num = sample_arr[..., self.temporal_known_num_indices].astype(float)
            # scale
            if self.scaling_method == 'mean_scaling':
                known_scale_arr = known_scale_arr.reshape(-1,1,len(self.temporal_known_num_indices))
                known_num = np.divide(known_num, known_scale_arr)
            elif self.scaling_method == 'standard_scaling':
                known_mean = known_scale_arr[:,0,:].reshape(-1,1,len(self.temporal_known_num_indices))
                known_stddev = known_scale_arr[:,1,:].reshape(-1,1,len(self.temporal_known_num_indices))
                known_num = np.divide(np.subtract(known_num, known_mean), known_stddev)
                known_num = np.nan_to_num(known_num)
            elif self.scaling_method == 'log_scaling':
                known_num = np.log1p(known_num)
            elif self.scaling_method == 'no_scaling':
                pass
            
            # merge
            model_in = np.concatenate((model_in, known_num), axis=-1)

        if len(self.temporal_unknown_num_indices) > 0:
            unknown_num = sample_arr[..., self.temporal_unknown_num_indices].astype(float)
            # scale
            if self.scaling_method == 'mean_scaling':
                unknown_scale_arr = unknown_scale_arr.reshape(-1,1,len(self.temporal_unknown_num_indices))
                unknown_num = np.divide(unknown_num, unknown_scale_arr)
            elif self.scaling_method == 'standard_scaling':
                unknown_mean = unknown_scale_arr[:,0,:].reshape(-1,1,len(self.temporal_unknown_num_indices))
                unknown_stddev = unknown_scale_arr[:,1,:].reshape(-1,1,len(self.temporal_unknown_num_indices))
                unknown_num = np.divide(np.subtract(unknown_num, unknown_mean), unknown_stddev)
                unknown_num = np.nan_to_num(unknown_num)
            elif self.scaling_method == 'log_scaling':
                unknown_num = np.log1p(unknown_num)
            elif self.scaling_method == 'no_scaling':
                pass
            
            # merge
            model_in = np.concatenate((model_in, unknown_num), axis=-1)
   
        if len(self.temporal_known_cat_indices) > 0:
            known_cat = sample_arr[..., self.temporal_known_cat_indices].astype(str)
            # merge
            model_in = np.concatenate((model_in, known_cat), axis=-1)

        if len(self.temporal_unknown_cat_indices) > 0:
            unknown_cat = sample_arr[..., self.temporal_unknown_cat_indices].astype(str)
            # merge
            model_in = np.concatenate((model_in, unknown_cat), axis=-1)
            
        rel_age = np.broadcast_to((np.arange(0,sequence_len,1)/sequence_len).reshape(-1,1), target.shape)
        model_in = np.concatenate((model_in, rel_age), axis=-1) 
        
        # scale
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling') or (self.scaling_method == 'log_scaling'):
            scale_in = np.broadcast_to(target_nz_mean, target.shape)
        elif self.scaling_method == 'standard_scaling':
            scale_mean = np.broadcast_to(target_mean, target.shape)
            scale_std = np.broadcast_to(target_stddev, target.shape)
            scale_in = np.concatenate((scale_mean, scale_std), axis=-1)
            
        model_in = np.concatenate((model_in, scale_in), axis=-1)
        
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling') or (self.scaling_method == 'log_scaling'):
            scale_out = np.broadcast_to(target_nz_mean, model_out.shape)
        elif self.scaling_method == 'standard_scaling':
            scale_mean_out = np.broadcast_to(target_mean, model_out.shape)
            scale_std_out = np.broadcast_to(target_stddev, model_out.shape)
            scale_out = np.concatenate((scale_mean_out, scale_std_out), axis=-1)
            
        mask_list = []
        for pad_len in pad_arr:
            if pad_len > 0:
                mask = np.concatenate((-1*np.ones((1,pad_len)), np.ones((1,sequence_len-pad_len))), axis=1)
            else:
                mask = np.ones((1,sequence_len))
            mask_list.append(mask)

        # mask & relative age
        mask = np.vstack(mask_list).reshape(-1,sequence_len,1)
        model_in = np.concatenate((model_in, mask), axis=-1)

        # sample weights
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling') or (self.scaling_method == 'log_scaling'):
            weights = np.around(np.log10(np.squeeze(target_nz_mean) + 10),2) #/np.quantile(np.squeeze(target_nz_mean), q=0.8)
        elif self.scaling_method == 'standard_scaling':
            weights = np.around(np.log10(np.squeeze(target_mean) + 10),2)
        weights = np.clip(weights, a_min=1.0, a_max=2.0)
        weights = weights.reshape(-1,1)
        
        return model_in, model_out, scale_out, weights
    
    def split_train_test(self, data):
        train_data = data[data[self.time_index_col]<=self.train_till].reset_index(drop=True)
        if self.datetime_format is not None:
            delta = max(self.train_test_timedelta, self.window_len) 
            test_data = data[data[self.time_index_col]<=self.test_till].groupby(self.id_col).apply(lambda x: x[-delta:]).reset_index(drop=True)
        else:
            # check if test_till - train_till > window_len
            test_len = int(data[(data[self.time_index_col]>self.train_till) & (data[self.time_index_col]<=self.test_till)].groupby(self.id_col)[self.target_col].count().max())
            test_len = test_len + (self.window_len - self.fh)
            test_data = data[data[self.time_index_col]<=self.test_till].groupby(self.id_col).apply(lambda x: x[-test_len:]).reset_index(drop=True)
            #test_data = data[data[self.time_index_col]<=self.test_till].groupby(self.id_col).apply(lambda x: x[-self.window_len:]).reset_index(drop=True) #original
        return train_data, test_data
    
    def train_test_dataset(self, data, train_till, test_till):
        
        self.train_till = train_till
        self.test_till = test_till
            
        # check null
        null_status, null_cols = self.check_null(data)
        if null_status:
            print("NaN column(s): ", null_cols)
            raise ValueError("Column(s) with NaN detected!")
        
        # sort
        data = self.sort_dataset(data)
        
        # check samples
        check_gen = self.data_generator(data=data, mode='train')
        x, y, s, w = next(check_gen)
        num_features = x.shape[-1]
        scale_dims = s.shape[-1]
        
        trainset = tf.data.Dataset.from_generator(lambda: self.data_generator(data=data, mode='train'),
                                                      output_signature=(tf.TensorSpec(shape=(None, self.window_len, num_features), dtype=tf.string),
                                                                        tf.TensorSpec(shape=(None, self.fh, 1), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(None, self.fh, scale_dims), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)))
        
        testset = tf.data.Dataset.from_generator(lambda: self.data_generator(data=data, mode='test'),
                                                     output_signature=(tf.TensorSpec(shape=(None, self.window_len, num_features), dtype=tf.string),
                                                                       tf.TensorSpec(shape=(None, self.fh, 1), dtype=tf.float32),
                                                                       tf.TensorSpec(shape=(None, self.fh, scale_dims), dtype=tf.float32),
                                                                       tf.TensorSpec(shape=(None, 1), dtype=tf.float32)))

        return trainset, testset
        

    def infer_dataset(self, data, history_till, future_till):
        self.history_till = history_till
        self.future_till = future_till
        
        # check null
        null_status, null_cols = self.check_null(data)
        if null_status:
            print("NaN column(s): ", null_cols)
            raise ValueError("Column(s) with NaN detected!")
            
        # sort & filter to recent context
        data = self.sort_dataset(data)
        #data = data[data[self.time_index_col]<=self.future_till].groupby(self.id_col).apply(lambda x: x[-self.window_len:]).reset_index(drop=True)

        arr, pad_arr, id_arr, date_arr, scale_arr, known_scale_arr, unknown_scale_arr = self.select_all_arrs(data)
        
        model_in, model_out, scale, _ = self.preprocess(arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr, mode='infer')
        input_tensor = tf.convert_to_tensor(model_in.astype(str), dtype=tf.string)
        
        # for evaluation
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling') or (self.scaling_method == 'log_scaling'):
            actuals_arr = model_out*scale
        elif self.scaling_method == 'standard_scaling':
            actuals_arr = model_out*scale[:,:,1:2] + scale[:,:,0:1]
            
        if len(self.static_cat_col_list)>0:
            stat_arr_list = []
            for col, i in zip(self.static_cat_col_list, self.static_cat_indices):
                stat_arr = model_in[:,-1:,i].astype(str)
                stat_arr_list.append(stat_arr)    
            stat_arr = np.concatenate(stat_arr_list, axis=1)        
            actuals_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1), stat_arr, actuals_arr.reshape(-1,self.fh)), axis=1))
            actuals_columns = ['id'] + self.static_cat_col_list + ['actual_{}'.format(i) for i in range(self.fh)]
            id_columns = ['id'] + self.static_cat_col_list
            actuals_df.columns = actuals_columns
            actuals_df = actuals_df.melt(id_vars=id_columns, value_name='actual').sort_values(id_columns).drop(columns=['variable'])
            actuals_df = actuals_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
            # merge with forecast period dates
            date_df = pd.DataFrame(date_arr.reshape(-1,)).rename(columns={0:'period'})
            actuals_df = pd.concat([actuals_df, date_df], axis=1)
            actuals_df['actual'] = actuals_df['actual'].astype(np.float32)
        else:
            actuals_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1), actuals_arr.reshape(-1,self.fh)), axis=1))
            actuals_columns = ['id'] + ['actual_{}'.format(i) for i in range(self.fh)]
            id_columns = ['id']
            actuals_df.columns = actuals_columns
            actuals_df = actuals_df.melt(id_vars=id_columns, value_name='actual').sort_values(id_columns).drop(columns=['variable'])
            actuals_df = actuals_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
            # merge with forecast period dates
            date_df = pd.DataFrame(date_arr.reshape(-1,)).rename(columns={0:'period'})
            actuals_df = pd.concat([actuals_df, date_df], axis=1)
            actuals_df['actual'] = actuals_df['actual'].astype(np.float32)
         
        return [input_tensor, scale, id_arr, date_arr], actuals_df
    
    def baseline_infer_dataset(self, data, history_till, future_till, ignore_cols, ignore_pad_values):
        self.history_till = history_till
        self.future_till = future_till
        
        # check null
        null_status, null_cols = self.check_null(data)
        if null_status:
            print("NaN column(s): ", null_cols)
            raise ValueError("Column(s) with NaN detected!")
            
        # sort & filter to recent context
        data = self.sort_dataset(data)
        #data = data[data[self.time_index_col]<=self.future_till].groupby(self.id_col).apply(lambda x: x[-self.window_len:]).reset_index(drop=True)
        
        # mask out columns in ignore_cols for baseline forecast
        for col,val in zip(ignore_cols, ignore_pad_values):
            data[col] = np.where(data[self.time_index_col]>self.history_till, val, data[col])
        
        arr, pad_arr, id_arr, date_arr, scale_arr, known_scale_arr, unknown_scale_arr = self.select_all_arrs(data)
     
        model_in, model_out, scale, _ = self.preprocess(arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr, mode='infer')
        input_tensor = tf.convert_to_tensor(model_in.astype(str), dtype=tf.string)
        
        return [input_tensor, scale, id_arr, date_arr]
    
    def acf(self, series):
        # https://stackoverflow.com/questions/43344406/is-there-a-bokeh-version-of-pandas-autocorrelation-plot-method
        n = len(series)
        data = np.asarray(series)
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / float(n)
        def r(h):
            acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
            return round(acf_lag, 3)
        x = np.arange(n) # Avoiding lag 0 calculation
        acf_coeffs = pd.Series(map(r, x)).round(decimals = 3)
        acf_coeffs = acf_coeffs + 0
        return acf_coeffs

    def significance(self, series):
        # https://stackoverflow.com/questions/43344406/is-there-a-bokeh-version-of-pandas-autocorrelation-plot-method
        n = len(series)
        z95 = 1.959963984540054 / np.sqrt(n)
        z99 = 2.5758293035489004 / np.sqrt(n)
        return (z95,z99)
    
    def show_ts_samples(self, data, sample_ids=[], n_samples=10, n_col=2, plot_size=(300,600), save=True, filename='ts_samples.html'):
        
        # sort dataset
        data = self.sort_dataset(data)
        
        num_covar_cols = self.col_dict.get('temporal_known_num_col_list') + self.col_dict.get('temporal_unknown_num_col_list')
        cat_covar_cols = self.col_dict.get('temporal_known_cat_col_list') + self.col_dict.get('temporal_unknown_cat_col_list')
        target_col = self.col_dict.get('target_col')
        id_col = self.col_dict.get('id_col')
        date_col = self.col_dict.get('time_index_col')
        
        # bokeh initialization
        reset_output()
        output_notebook()
        TOOLS = "box_select,lasso_select,xpan,reset,save"
        h,w = plot_size
        
        if save:
            output_file(filename)
        
        if len(sample_ids) == 0:    
            # randomly sample from id_cols
            keys_dict, wts_dict = self.get_keys(data)
            sample_ids = []
            for k,v in keys_dict.items():
                ids = list(np.random.choice(v, size=n_samples))
                sample_ids += ids
        else:
            # use provided ids for sampling
            pass
        
        assert len(sample_ids) > 0,  "At least one Key required!"
        
        saveplots = []
        for sid in sample_ids:
            df_sample = data[data[id_col]==sid]
            df_sample = df_sample.reset_index()
            df_sample = df_sample[[target_col, date_col] + num_covar_cols + cat_covar_cols]
            df_sample[date_col] = df_sample[date_col].astype(str)
            num_cols = [target_col] + num_covar_cols
            num_columns = len(num_cols)
            source = ColumnDataSource(data=df_sample)
            plots = []
            for col in num_cols:
                y_vals = df_sample[col].unique().tolist()
                p = figure( plot_height=h, plot_width=w, tools=TOOLS, x_axis_label='timestep', y_axis_label=col, title = "{} {}".format(sid,col))
                p.line(x='index', y=col, source=source)
                tooltips = [(col,'@{}'.format(col)), (date_col,'@{}'.format(date_col))]
                p.add_tools(HoverTool(tooltips=tooltips))
                plots.append(p)
            
            # autocorr plot
            series = df_sample[target_col]
            x = pd.Series(range(1, len(series)+1), dtype = float)
            z95, z99 = self.significance(series)
            y = self.acf(series)
            p = figure(title="{} Auto-Correlation Plot".format(sid), plot_height=h, plot_width=w, x_axis_label="Lag", y_axis_label="Autocorrelation")
            p.line(x, z99, line_dash='dashed', line_color='grey')
            p.line(x, z95, line_color = 'grey')
            p.line(x, y=0.0, line_color='black')
            p.line(x, z99*-1, line_dash='dashed', line_color='grey')
            p.line(x, z95*-1, line_color = 'grey')
            p.line(x, y, line_width=2)
            plots.append(p)
    
            for i in range(len(plots)):
                if i==0:
                    pass
                else:
                    plots[i].x_range = plots[i-1].x_range
        
            subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]
            saveplots += subplots
            
            if len(cat_covar_cols)>0:
                plots = []
                for col in cat_covar_cols:
                    df_cat = df_sample.groupby(col).agg({target_col:'mean'}).reset_index()
                    source = ColumnDataSource(data=df_cat)
                    p = figure(x_range=df_cat[col].astype(str).unique(), plot_height=h, plot_width=w, tools=TOOLS, x_axis_label=col, y_axis_label='Mean ' + str(target_col), title = "{} {}".format(sid,col))
                    p.vbar(x=col, top=target_col, source=source, width=0.8)
                    p.xaxis.major_label_orientation = "vertical"
                    tooltips = [(col,'@{}'.format(col)),(target_col,'@{}'.format(target_col))]
                    p.add_tools(HoverTool(tooltips=tooltips))
                    plots.append(p)
        
                subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]
                saveplots += subplots
                html = """<h3>{}: Time Series Plots with Numerical Covariates, Mean by Categorical Covariates</h3>""".format(target_col)
                sup_title = Div(text=html)    
            else:
                html = """<h3>{}: Time Series Plots with Numerical Covariates</h3>""".format(target_col)
                sup_title = Div(text=html)
                
        grid = gridplot(saveplots)
        show(column(sup_title, grid))
        
    def show_processed_ts_samples(self, data, n_samples=10, n_col=2, plot_size=(300,400), save=True, filename='processed_ts_samples.html'):
        # sort
        data = self.sort_dataset(data)
        
        # Obtain col names & col positions in the input tensor
        target_col_name, target_index = self.col_index_dict.get('target_index')
        stat_cat_col_names, stat_cat_indices = self.col_index_dict.get('static_cat_indices')
        known_num_col_names, known_num_indices = self.col_index_dict.get('temporal_known_num_indices')
        unknown_num_col_names, unknown_num_indices = self.col_index_dict.get('temporal_unknown_num_indices')
    
        # get samples
        sample_gen = self.data_generator(data)
        x, y, s, w = next(sample_gen)
        
        # bokeh initialization
        reset_output()
        output_notebook()
        TOOLS = "box_select,lasso_select,xpan,reset,save"
        h,w = plot_size
        x_range = np.arange(self.window_len).tolist() 
        
        if save:
            output_file(filename)
            
        # display sample series
        layouts = []
        for i in range(len(x)):
            
            if i >= n_samples:
                    break
                    
            sid = str(x[i,-1,0])             # shape: [timsteps, features]
            scale = str(s[i,-1,0])
            target = x[i,:,1].tolist()
            mask = x[i,:,-1].tolist()
            
            # static cat columns
            stat_cat_cols = []
            stat_cat_cols.append(sid)
            stat_cat_cols.append(scale)
            if len(stat_cat_indices)>0:
                for col, k in zip(stat_cat_col_names, stat_cat_indices):
                    stat_cat_cols.append(str(x[i,-1,k]))
                    
            # temporal num columns
            temp_num_cols = []
            temp_num_cols.append(target)
            temp_num_cols.append(mask)
            if len(known_num_indices + unknown_num_indices)>0:
                for col, k in zip(known_num_col_names + unknown_num_col_names, known_num_indices + unknown_num_indices):
                    temp_num_cols.append(x[i,:,k].tolist())
            
            num_col_names = [self.target_col] + ['mask'] + known_num_col_names + unknown_num_col_names
            stat_val = '_'.join(stat_cat_cols)
            
            plots = []
            for j, col in enumerate(num_col_names):
                p = figure(plot_height=h, plot_width=w, tools=TOOLS, x_axis_label='timestep', y_axis_label=col, title = "{}".format(stat_val))
                p.line(x=x_range, y=temp_num_cols[j])
                tooltips = [(col,'@{}'.format(col))]
                p.add_tools(HoverTool(tooltips=tooltips))
                plots.append(p)
    
            for i in range(len(plots)):
                if i==0:
                    pass
                else:
                    plots[i].x_range = plots[i-1].x_range
        
            subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]
            html = """<h3>{}: Scaled Time Series Plots (Numerical Columns)</h3>""".format(sid)
            sup_title = Div(text=html)
            grid = gridplot(subplots)
            layout = column(sup_title, grid, sizing_mode="stretch_both")
            layouts.append(layout)
            i += 1
        
        supergrid = gridplot(layouts, ncols=1)
        show(supergrid)




