# Databricks notebook source
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
from sys import getsizeof
import psutil

class sage_dataset:
    def __init__(self, 
                 col_dict, 
                 window_len, 
                 fh, 
                 batch = 128, 
                 min_nz = 1,
                 max_per_key_train_samples = -1,
                 max_per_key_test_samples = -1,
                 scaling_method = 'mean_scaling',
                 interleave = 1, 
                 PARALLEL_DATA_JOBS = 4, 
                 PARALLEL_DATA_JOBS_BATCHSIZE = 128, 
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
        self.max_key_train_samples = max_per_key_train_samples
        self.max_key_test_samples = max_per_key_test_samples
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
        size = len(v)
            
        return list(np.random.choice(v, size=size, replace=False, p=w))
    
    def select_ids(self, keys_dict, wts_dict):
        """
        Sample Ids for a train/test iteration from the Keys_dict
        """
        n_jobs = min(len(keys_dict),4)
        sample_id = Parallel(n_jobs=n_jobs, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(self.dict_sampler)(v1,v2) for (k1,v1),(k2,v2) in zip(keys_dict.items(), wts_dict.items()) )
        sample_id = list(itertools.chain.from_iterable(sample_id))
        
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

        arr = np.vstack(arr_list)
        pad_arr = np.vstack(pad_list)
        scale_arr = np.vstack(scale_list)
        known_scale_arr = np.vstack(known_scale_list)
        unknown_scale_arr = np.vstack(unknown_scale_list)

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

        num_samples = max(1, delta+1)

        if mode == 'train':
          max_samples = num_samples if self.max_key_train_samples == -1 else self.max_key_train_samples
        elif mode == 'test':
          max_samples = num_samples if self.max_key_test_samples == -1 else self.max_key_test_samples
        
        arr_list = []
        pad_list = []
        scale_list = []
        known_scale_list = []
        unknown_scale_list = []

        if delta<0:
            arr = gdf.loc[:,self.col_list].reset_index(drop=True).values
            # left pad to window_len with 0
            pad_len = np.abs(delta)
            arr = np.pad(arr,pad_width=((pad_len,0),(0,0)), mode='constant', constant_values=0)
            arr_list.append(arr)
            pad_list.append([pad_len])
            scale_list.append(scale)
            known_scale_list.append(known_scale)
            unknown_scale_list.append(unknown_scale)
        elif delta==0:
            pad_len = 0
            arr = gdf.loc[0:self.window_len - 1, self.col_list].reset_index(drop=True).values
            arr_list.append(arr)
            pad_list.append([pad_len])
            scale_list.append(scale)
            known_scale_list.append(known_scale)
            unknown_scale_list.append(unknown_scale)
        else:
            pad_len = 0
            sample_interval = max(1, int(num_samples/max_samples))
            # take only the most recent max_samples
            start_pos = max(0, int(len(gdf) - (sample_interval * (max_samples - 1) + self.window_len)))

            for i in range(start_pos, num_samples, sample_interval):
              arr = gdf.loc[i:i + self.window_len - 1, self.col_list].reset_index(drop=True).values
              arr_list.append(arr)
              pad_list.append([pad_len])
              scale_list.append(scale)
              known_scale_list.append(known_scale)
              unknown_scale_list.append(unknown_scale)
        
        arr = np.stack(arr_list, axis=0)
        pad_arr = np.stack(pad_list, axis=0)
        scale_arr = np.stack(scale_list, axis=0)
        known_scale_arr = np.stack(known_scale_list, axis=0)
        unknown_scale_arr = np.stack(unknown_scale_list, axis=0)

        return (arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr)
    
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
        else:
            pad_len = 0
            arr = gdf.loc[0:self.window_len - 1, self.col_list].reset_index(drop=True).values
            date_arr = gdf.loc[0:self.window_len - 1, self.time_index_col].reset_index(drop=True).tail(self.fh).values
        
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
        #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

        keys_dict, wts_dict = self.get_keys(data)
        all_id = self.select_ids(keys_dict, wts_dict)
        all_id.sort()
        num_ids = len(all_id)

        i = 0
        while (i+1)*self.batch < num_ids+self.batch:
          sample_id = all_id[i*self.batch:(i+1)*self.batch]
          df = data.query("{}==@sample_id".format(self.id_col))
          arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr = self.select_arrs(df, mode)
          #print(arr.shape, pad_arr.shape, scale_arr.shape, known_scale_arr.shape, unknown_scale_arr.shape)
          #print(getsizeof(arr), getsizeof(pad_arr), getsizeof(scale_arr), getsizeof(known_scale_arr), getsizeof(unknown_scale_arr))

          #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

          #print("data gen: ", scale_arr.shape)
          # -- done for @tf.function retracing reason. May revisit later
          valid_indices = np.where(np.count_nonzero(arr[:,:self.window_len-self.fh,self.target_index].astype(np.float32)>0,axis=1)>=self.min_nz)
          arr = arr[valid_indices]
          pad_arr = pad_arr[valid_indices]
          scale_arr = scale_arr[valid_indices]
          known_scale_arr = known_scale_arr[valid_indices]
          unknown_scale_arr = unknown_scale_arr[valid_indices]

          #print("processing ...")
          model_in, model_out, scale, wts  = self.preprocess(arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr, mode)

          #print("done processing ...")  
          #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

          i += 1

          yield model_in.astype(str), model_out.astype(np.float32), scale.astype(np.float32), wts.astype(np.float32)
            
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
        
        # scale target : target/target_mean
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling'):
          sample_arr[..., [self.target_index]] = np.divide(sample_arr[..., [self.target_index]].astype(float), scale_arr.reshape(-1,1,1))
        elif self.scaling_method == 'standard_scaling':
          sample_arr[..., [self.target_index]] = np.nan_to_num(np.divide(np.subtract(sample_arr[..., [self.target_index]].astype(float), scale_arr[:,0].reshape(-1,1,1)), scale_arr[:,1].reshape(-1,1,1)))

        if mode == 'infer':
          sample_arr[:, -self.fh:, self.target_index] = 0
            
        if len(self.temporal_known_num_indices) > 0:
            if self.scaling_method == 'mean_scaling':
                sample_arr[..., self.temporal_known_num_indices] = np.divide(sample_arr[..., self.temporal_known_num_indices].astype(float), known_scale_arr.reshape(-1,1,len(self.temporal_known_num_indices)))
            elif self.scaling_method == 'standard_scaling':
                sample_arr[..., self.temporal_known_num_indices] = np.divide(np.subtract(sample_arr[..., self.temporal_known_num_indices].astype(float), known_scale_arr[:,0,:].reshape(-1,1,len(self.temporal_known_num_indices))), known_scale_arr[:,1,:].reshape(-1,1,len(self.temporal_known_num_indices)))
            elif self.scaling_method == 'no_scaling':
                pass
            
        if len(self.temporal_unknown_num_indices) > 0:
            if self.scaling_method == 'mean_scaling':
                sample_arr[..., self.temporal_unknown_num_indices] = np.divide(sample_arr[..., self.temporal_unknown_num_indices].astype(float), unknown_scale_arr.reshape(-1,1,len(self.temporal_unknown_num_indices)))
            elif self.scaling_method == 'standard_scaling':
                sample_arr[..., self.temporal_unknown_num_indices] = np.divide(np.subtract(sample_arr[..., self.temporal_unknown_num_indices].astype(float), unknown_scale_arr[:,0,:].reshape(-1,1,len(self.temporal_unknown_num_indices))), unknown_scale_arr[:,1,:].reshape(-1,1,len(self.temporal_unknown_num_indices)))
            elif self.scaling_method == 'no_scaling':
                pass
        
        # relative age of datapoints
        rel_age = np.broadcast_to((np.arange(0,sequence_len,1)/sequence_len).reshape(-1,1), sample_arr[..., [self.target_index]].shape)

        # scale-in
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling'):
            scale_in = np.broadcast_to(scale_arr.reshape(-1,1,1), sample_arr[..., [self.target_index]].shape)
        elif self.scaling_method == 'standard_scaling':
            scale_mean = np.broadcast_to(scale_arr[:,0].reshape(-1,1,1), sample_arr[..., [self.target_index]].shape)
            scale_std = np.broadcast_to(scale_arr[:,1].reshape(-1,1,1), sample_arr[..., [self.target_index]].shape)
            scale_in = np.concatenate((scale_mean, scale_std), axis=-1)
        
        #scale-out
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling'):
          scale_out = np.broadcast_to(scale_arr.reshape(-1,1,1), sample_arr[:, -self.fh:, [self.target_index]].shape)
        elif self.scaling_method == 'standard_scaling':
          scale_mean_out = np.broadcast_to(scale_arr[:,0].reshape(-1,1,1), sample_arr[:, -self.fh:, [self.target_index]].shape)
          scale_std_out = np.broadcast_to(scale_arr[:,1].reshape(-1,1,1), sample_arr[:, -self.fh:, [self.target_index]].shape)
          scale_out = np.concatenate((scale_mean_out, scale_std_out), axis=-1)

        # mask indicator for lack of history    
        mask = np.broadcast_to(pad_arr.reshape(-1,1,1), (pad_arr.shape[0],sequence_len,1))
        mask_indices = np.indices(mask.shape)[1]
        mask = np.where(mask>mask_indices, -1, 1)

        sample_arr = np.concatenate((sample_arr, rel_age, scale_in, mask), axis=-1)
        model_out = sample_arr[:, -self.fh:, [self.target_index]].astype(float)

        # sample weights
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling'):
          weights = np.around(np.log1p(np.squeeze(scale_arr.reshape(-1,1,1))), 2)
        elif self.scaling_method == 'standard_scaling':
          weights = np.around(np.log1p(np.squeeze(scale_arr[:,0].reshape(-1,1,1))), 2)
        weights = np.clip(weights, a_min=0.0, a_max=10.0)
        weights = weights.reshape(-1,1)

        #print("merge done")
        #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

        del rel_age, mask, pad_arr, mask_indices, scale_arr, scale_in
        gc.collect()
        
        return sample_arr, model_out, scale_out, weights
    
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
        
        #print("start ... ")
        #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

        scale_dims = 1 if self.scaling_method in ['mean_scaling', 'no_scaling'] else 2
        num_features = len(self.col_list) + scale_dims + 2  # rel_age, scale_in, mask

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
       
        arr, pad_arr, id_arr, date_arr, scale_arr, known_scale_arr, unknown_scale_arr = self.select_all_arrs(data)
        
        model_in, model_out, scale, _ = self.preprocess(arr, pad_arr, scale_arr, known_scale_arr, unknown_scale_arr, mode='infer')
        input_tensor = tf.convert_to_tensor(model_in.astype(str), dtype=tf.string)
        
        # for evaluation
        if (self.scaling_method == 'mean_scaling') or (self.scaling_method == 'no_scaling'):
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
    


