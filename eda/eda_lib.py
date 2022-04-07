#!/usr/bin/env python
# coding: utf-8

# basic imports
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
import scipy
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from statsmodels.tsa.stattools import adfuller

# bokeh imports
from bokeh.plotting import figure, output_file, show, output_notebook, save
from bokeh.models import ColumnDataSource, HoverTool, Div, FactorRange
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Category10, Category20, Colorblind
from bokeh.io import reset_output
from bokeh.models.ranges import DataRange1d

# holoviews import
import holoviews as hv
from holoviews import opts,dim
import hvplot.pandas
import hvplot.dask
hv.extension('bokeh')

# ennemi import
from ennemi import estimate_mi, normalize_mi, pairwise_mi

class eda:
    def __init__(self, col_dict, PARALLEL_DATA_JOBS=4, PARALLEL_DATA_JOBS_BATCHSIZE=128):
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

        """
        self.col_dict = col_dict
        self.PARALLEL_DATA_JOBS = PARALLEL_DATA_JOBS
        self.PARALLEL_DATA_JOBS_BATCHSIZE = PARALLEL_DATA_JOBS_BATCHSIZE
        
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
    
    def sort_dataset(self, data):
        """
        sort pandas dataframe by provided col list & order
        """
        if len(self.sort_col_list) > 0:
            data = data.sort_values(by=self.sort_col_list, ascending=True)
        else:
            pass
        return data
    
    def create_report(self, data, time_lags=[-1,0,1], filename='eda.html', n_col=2, plot_size=(400,800), max_static_col_levels=100):
        # prepare static cols list
        stat_cols = self.static_num_col_list + self.static_cat_col_list
        if len(stat_cols)==0:
            # add a dummy column 'dummy_static_col'
            data['dummy_static_col'] = 'dummy_static_col'
            stat_cols = ['dummy_static_col']
        target_col = self.target_col
        id_col = self.id_col
        date_col = self.time_index_col
        
        # Sort Dataset
        data = self.sort_dataset(data)

        # sort in ascending order stat_cols by unique count
        stat_counts = {}
        for col in stat_cols:
            stat_counts[col] = data[col].nunique()
        stat_counts = {k: v for k, v in sorted(stat_counts.items(), key=lambda item: item[1])}

        # filter out stat_cols with more than max_static_col_levels levels
        stat_counts = {k: v for k, v in stat_counts.items() if v<=max_static_col_levels}
        stat_cols = list(stat_counts.keys())
        
        # bokeh initialization
        reset_output()
        output_notebook()
        TOOLS = "box_select,lasso_select,xpan,reset,save"
        h,w = plot_size
        
        # save
        output_file(filename)
        
        # Key counts by static cols
        plots = []
        for stat_col in stat_cols:
                title = 'Count of keys by ' + stat_col
                df_cat = data.groupby([stat_col]).agg({id_col: lambda x: x.nunique()}).reset_index()
                bars = hv.Bars(
                            data = df_cat,
                            kdims = [stat_col],
                            vdims = id_col,
                            label = title
                            ).opts(
                             show_legend=False,
                             height=plot_size[0], 
                             width=plot_size[1],
                             xrotation=90,
                            ).opts(
                             tools=['hover'], 
                            )
                fig = hv.render(bars)
                plots.append(fig)

        html = """<h3>Key Counts by Static Columns</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_1 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # multilevel counts
        stat_col_groups = list(itertools.combinations(stat_cols,2))
        plots = []
        for col_grp in stat_col_groups:
            col_grp = list(col_grp)
            title = 'Count of Keys by ' + col_grp[0] + ',' + col_grp[1]
            df_cat = data.groupby(col_grp).agg({id_col: lambda x: x.nunique()}).reset_index()
            bars = hv.Bars(
                data = df_cat,
                kdims = col_grp,
                vdims = id_col,
                label = title
                ).opts(
                show_legend=False,
                height=plot_size[0], 
                width=plot_size[1],
                xrotation=90,
                ).opts(
                tools=['hover']
                )
            fig = hv.render(bars)
            plots.append(fig)
        
        html = """<h3>Key Counts by Static Columns Combinations</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_2 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # target total volume distribution by stat cols
        plots = []
        for stat_col in stat_cols:
            title = 'Distribution of sum of ' + target_col + ' by ' + stat_col
            df_cat = data.groupby([stat_col]).agg({target_col:'sum'}).reset_index()
            bars = hv.Bars(
                        data = df_cat,
                        kdims = [stat_col],
                        vdims = target_col,
                        label = title
                        ).opts(
                         show_legend=False,
                         height=plot_size[0], 
                         width=plot_size[1],
                         xrotation=90,
                        ).opts(
                         tools=['hover'], 
                        )
            fig = hv.render(bars)
            plots.append(fig)
        
        html = """<h3>Target Total Distribution by Static Columns</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_3 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # Basic Stats boxplot
        plots = []
        for stat_col in stat_cols:
            title = 'Box & Whisker plot of basic stats of ' + target_col + ' by ' + stat_col
            df_stat = data.groupby([stat_col, id_col]).agg({target_col:['mean','median','std']})
            df_stat = df_stat.droplevel(0, axis=1).reset_index()
            df_stat = pd.melt(df_stat[[stat_col,'mean','median','std']], id_vars=[stat_col], value_vars=['mean','std','median'])
            df_stat.rename(columns={'value':target_col, 'variable':'statistic'}, inplace=True)
            boxwhisker = hv.BoxWhisker(
                                       data = df_stat, 
                                       kdims = [stat_col, 'statistic'], 
                                       vdims = target_col, 
                                       label = title
                                      ).opts(
                                       show_legend=False,
                                       height=plot_size[0], 
                                       width=plot_size[1],
                                       xrotation=90,
                                       box_fill_color=dim('statistic').str(), 
                                       cmap='Set1'
                                      ).opts(
                                       tools=['hover'], 
                                      )

            fig = hv.render(boxwhisker)
            plots.append(fig)
        
        html = """<h3>Distribution of Basic Statistics as BoxWhiskers plot</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_4 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # CoV Countplot
        plots = []
        for stat_col in stat_cols:
            title = 'Key Counts by CoV by ' + stat_col
            df_stat = data.groupby([stat_col,id_col]).agg({target_col:['mean','median','std']})
            df_stat = df_stat.droplevel(0, axis=1).reset_index()
            df_stat['cov'] = df_stat['std']/df_stat['mean']
            hist = df_stat.hvplot.hist(y='cov', by=stat_col, bins=20, title=title, height=plot_size[0], width=plot_size[1], legend='top')
            fig = hv.render(hist)
            plots.append(fig)
        
        html = """<h3>Distribution of Keys by CoV</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_5 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # CoV evolution
        plots = []
        for stat_col in stat_cols:
            title = 'CoV evolution over time by ' + stat_col
            data['cov_period'] = pd.qcut(data[date_col], q=4, labels=['p25','p50','p75','p100'])
            df_stat = data.groupby(['cov_period',stat_col,id_col]).agg({target_col:['mean','median','std']})
            df_stat = df_stat.droplevel(0, axis=1).reset_index()
            df_stat['cov'] = df_stat['std']/df_stat['mean']
            hist = df_stat.hvplot.hist(y='cov', ylabel='Count', by=['cov_period',stat_col], bins=20, title=title, legend='top', height=plot_size[0], width=plot_size[1])
            fig = hv.render(hist)
            plots.append(fig)
        
        html = """<h3>Distribution of Keys by CoV over period quantiles - p25 (1st qtr), p50 (2nd qtr), ...</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_6 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # CoV volume plot
        plots = []
        for stat_col in stat_cols:
                title = 'Volume by CoV by ' + stat_col
                df_stat = data.groupby([stat_col,id_col]).agg({target_col:['sum','mean','median','std']})
                df_stat = df_stat.droplevel(0, axis=1).reset_index()
                df_stat['cov'] = df_stat['std']/df_stat['mean']
                bins = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 10.0, 1000.0]
                df_stat['cov_range'] = pd.cut(df_stat['cov'], bins=bins, include_lowest=True)
                df_stat['cov_range'] = df_stat['cov_range'].astype(str)
                df_stat = df_stat.groupby([stat_col,'cov_range']).agg({'sum':'sum','mean':'mean'}).reset_index()
                hist = df_stat.hvplot.bar(x=stat_col, y=['sum'], by='cov_range', rot=90, title=title, legend='top', height=plot_size[0], width=plot_size[1])
                fig = hv.render(hist)
                plots.append(fig)
        
        html = """<h3>Distribution of Volume by CoV </h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_7 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # Data Availability
        plots = []
        for stat_col in stat_cols:
            title = "Count no. of datapoints by " + stat_col
            df_stat = data.groupby([stat_col,id_col]).agg({target_col:['count','sum']})
            df_stat = df_stat.droplevel(0, axis=1).reset_index()
            hist = df_stat.hvplot.hist(y='count', by=[stat_col], bins=10, title=title, height=plot_size[0], width=plot_size[1], legend='top')
            fig = hv.render(hist)
            plots.append(fig)
        
        html = """<h3>Distribution of Datapoints</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_8 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # Intermittence Plots
        # NonZero Count fn.
        def nz_percent(x):
            nz_perc = np.count_nonzero(x)/len(x)*100
            return nz_perc

        plots = []
        for stat_col in stat_cols:
            title = "Count no. of nonzero datapoints by " + stat_col
            df_stat = data.groupby([stat_col,id_col]).agg({target_col:[nz_percent]})
            df_stat = df_stat.droplevel(0, axis=1).reset_index()
            hist = df_stat.hvplot.hist(y='nz_percent', by=[stat_col], bins=10, title=title, height=plot_size[0], width=plot_size[1], legend='top')
            fig = hv.render(hist)
            plots.append(fig)
        
        html = """<h3>Distribution of Non-Zero Datapoints</h3>"""
        sup_title = Div(text=html)
        subplots = [plots[i:i + n_col] for i in range(0, len(plots), n_col)]   
        grid = gridplot(subplots)
        layout_9 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # ts correlation plot (with various numerical & categorical temporal features)
        temp_num_cols = self.temporal_known_num_col_list + self.temporal_unknown_num_col_list
        temp_cat_cols = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list
        
        # stationarity test fn.
        def adf_test(x):
            for col in [target_col]+temp_num_cols:
                result = adfuller(x[col].values)
                adf_stat = result[0]
                p_val = result[1]
                critical_val = min(list(result[4].values()))
                if p_val > 0.01:
                    if adf_stat > critical_val:
                        x['{}_stationary'.format(col)] = 0
                    else:
                        # stationary
                        x['{}_stationary'.format(col)] = 1
                else:
                    # inconclusive -- treat as stationary
                    x['{}_stationary'.format(col)] = 1
            
            return x
        
        # calculate stationarity columns
        groups = data.groupby(id_col)
        adf_df = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(adf_test)(x) for _,x in groups)
        data = pd.concat(adf_df, axis=0)
        data = data.reset_index(drop=True)
        
        def pearson_coeff(x, lag, stat_col, id_col):
            # reduce temporal/auto correlation using differencing
            for col in [target_col]+temp_num_cols:
                if x['{}_stationary'.format(col)].max()==0:
                    x[col] = x[col].diff(periods=1)
                
            x = x.dropna(subset=[target_col]+temp_num_cols)
            
            x[temp_num_cols] = x[temp_num_cols].shift(periods=lag)
            x = x.dropna(subset=temp_num_cols)
            
            corr_df = x[[target_col]+temp_num_cols].corr()
            corr_df = corr_df.reset_index()
            corr_df.rename(columns={'index':'level_2'}, inplace=True)
    
            x = x[[stat_col, id_col]].drop_duplicates().reset_index(drop=True)
            corr_df = pd.concat([x, corr_df], axis=1)
            corr_df.fillna(method='ffill', inplace=True)

            return corr_df
            
        # Pearson Correlation Distribution
        plots = []
        for stat_col in stat_cols:
            df_stat = data.groupby([stat_col, id_col]).filter(lambda x: len(x) > max(10,(len(time_lags) + 6)))
            df_stat = df_stat.groupby([stat_col, id_col]).filter(lambda x: len(x[target_col].unique()) > 1)
            subplots = []
            for lag in time_lags:
                #df_corr = df_stat.groupby([stat_col, id_col]).apply(lambda x: pearson_coeff(x, lag)).reset_index()
                # parallel process
                groups = df_stat.groupby([stat_col, id_col])
                corr = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(pearson_coeff)(x, lag, stat_col, id_col) for _,x in groups)
                df_corr = pd.concat(corr, axis=0)
                df_corr = df_corr.reset_index(drop=True)
                for temp_num_col in temp_num_cols:
                    for level in df_corr[stat_col].unique().tolist():
                        title = "Corr. coeff density over " + stat_col +  " between " + target_col + " & " + temp_num_col
                        df_temp = df_corr[(df_corr[stat_col]==level)&(df_corr['level_2']==target_col)][[stat_col,id_col,target_col,temp_num_col]]
                        points = hv.Distribution(df_temp[temp_num_col].values).opts(xlabel=target_col + '_' + temp_num_col + '_' + str(lag) + '_corr_coeff',
                                                                                    ylabel="density_over_" + level,
                                                                                    height=plot_size[0], 
                                                                                    width=plot_size[1], 
                                                                                    title=title).opts(tools=['hover'],)              
                        fig = hv.render(points)
                        subplots.append(fig)
            plots.append(subplots)
        
        html = """<h3>Distribution of linear correlation coefficient</h3>"""
        sup_title = Div(text=html)
        grid = gridplot(plots)
        layout_10 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # MI specific functions
        rng = np.random.default_rng(1234) 
        
        def mi(x, stat_col, id_col):
            # reduce temporal/auto correlation using differencing
            for col in [target_col]+temp_num_cols:
                if x['{}_stationary'.format(col)].max()==0:
                    x[col] = x[col].diff(periods=1)
            
            x = x.dropna(subset=[target_col]+temp_num_cols)
            
            # Add random noise to avoid discrete/low resolution distributions. MI goes to -inf otherwise
            for col in temp_num_cols:
                x[col] += rng.normal(0, 1e-6, size=len(x))
            x[target_col] += rng.normal(0, 1e-6, size=len(x))
                
            # lags
            lags = time_lags
            emi = estimate_mi(x[target_col], x[temp_num_cols], lag=lags, normalize=True, preprocess=True)
            emi = emi.reset_index()
            emi.rename(columns={'index':'level_2'}, inplace=True)
            
            x = x[[stat_col, id_col]].drop_duplicates().reset_index(drop=True)
            emi = pd.concat([x, emi], axis=1)
            emi.fillna(method='ffill', inplace=True)
            
            return emi
       
        def mi_discrete(x, stat_col, id_col):
            # reduce temporal/auto correlation using differencing
            for col in [target_col]:
                if x['{}_stationary'.format(col)].max()==0:
                    x[col] = x[col].diff(periods=1)

            x = x.dropna(subset=[target_col] + temp_cat_cols)
            
            temp_cat_arr = x[temp_cat_cols].values
            x[target_col] += rng.normal(0, 1e-6, size=len(x))
            target_arr = x[target_col].values
            
            # lags
            lags = time_lags
            emi = estimate_mi(target_arr, temp_cat_arr, discrete_x=True, lag=lags, normalize=False)
            emi = normalize_mi(emi)
            emi_df = pd.DataFrame(emi)
            emi_df.columns = temp_cat_cols
            emi_df['level_2'] = pd.Series(lags) 
    
            x = x[[stat_col, id_col]].drop_duplicates().reset_index(drop=True)
            emi_df = pd.concat([x, emi_df], axis=1)
            emi_df.fillna(method='ffill', inplace=True)
            
            return emi_df
        
        # Non-linear Correlation (MI) Distribution - Numeric Columns
        plots = []  
        for stat_col in stat_cols: 
            df_stat = data.groupby([stat_col, id_col]).filter(lambda x: len(x) > max(10,(len(time_lags) + 6)))
            #df_corr = df_stat.groupby([stat_col, id_col]).apply(lambda x: mi(x)).reset_index()
            # parallel process
            groups = df_stat.groupby([stat_col, id_col])
            emi = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(mi)(x, stat_col, id_col) for _,x in groups)            
            df_corr = pd.concat(emi, axis=0)
            df_corr = df_corr.reset_index(drop=True)
            subplots = []
            for temp_num_col in temp_num_cols:
                df_temp = df_corr[[stat_col,id_col,temp_num_col,'level_2']]
                for level in df_temp[stat_col].unique().tolist():
                    for lag in time_lags:
                        title = "MI density over " + stat_col +  " between " + target_col + " & " + temp_num_col
                        df_level = df_temp[(df_temp[stat_col]==level)&(df_temp['level_2']==lag)]
                        points = hv.Distribution(df_level[temp_num_col].values).opts(xlabel=target_col + '_' + temp_num_col + '_lag_' + str(lag) + '_mi', 
                                                                                 ylabel='density_over_'+ level,
                                                                                 height=plot_size[0], 
                                                                                 width=plot_size[1],
                                                                                 title=title).opts(tools=['hover'],)

                        fig = hv.render(points)
                        subplots.append(fig)
            plots.append(subplots)
        
        html = """<h3> Non-linear correlation (measured as Mutual Information) plots for Numeric Columns.</h3>"""
        sup_title = Div(text=html)
        grid = gridplot(plots)
        layout_11 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # Non-linear Correlation (MI) Distribution - Non-Numeric Columns
        plots = [] 
        for stat_col in stat_cols: 
            df_stat = data.groupby([stat_col, id_col]).filter(lambda x: len(x) > max(10,(len(time_lags) + 6)))
            df_stat = df_stat.groupby([stat_col, id_col]).filter(lambda x: len(x[target_col].unique()) > 1)
            #df_corr = df_stat.groupby([stat_col, id_col]).apply(lambda x: mi_discrete(x)).reset_index()
            #df_corr['level_2'].replace(to_replace=df_corr['level_2'].unique().tolist(), value=time_lags, inplace=True)
            # parallel process
            groups = df_stat.groupby([stat_col, id_col])
            emi = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(mi_discrete)(x, stat_col, id_col) for _,x in groups)            
            df_corr = pd.concat(emi, axis=0)
            df_corr = df_corr.reset_index(drop=True)
            subplots = []
            for temp_cat_col in temp_cat_cols:
                df_temp = df_corr[[stat_col,id_col,temp_cat_col,'level_2']]
                for level in df_temp[stat_col].unique().tolist():
                    for lag in time_lags:
                        title = "MI density over " + stat_col +  " between " + target_col + " & " + temp_cat_col
                        df_level = df_temp[(df_temp[stat_col]==level)&(df_temp['level_2']==lag)]
                        points = hv.Distribution(df_level[temp_cat_col].values).opts(xlabel=target_col + '_' + temp_cat_col + '_lag_' + str(lag) + '_mi', 
                                                                                 ylabel='density_over_'+level,
                                                                                 height=plot_size[0], 
                                                                                 width=plot_size[1],
                                                                                 title=title).opts(tools=['hover'],)

                        fig = hv.render(points)
                        subplots.append(fig)
            plots.append(subplots)
        
        html = """<h3> Non-linear correlation (measured as Mutual Information) plots for Categorical Columns.</h3>"""
        sup_title = Div(text=html)
        grid = gridplot(plots)
        layout_12 = column(sup_title, grid, sizing_mode="stretch_both")
        
        # supergrid
        supergrid = gridplot([layout_1,layout_2,layout_3,layout_4,layout_5,layout_6,layout_7,layout_8,layout_9,layout_10,
                              layout_11,layout_12], ncols=1)
        save(supergrid)
        
    def get_correlations(self, data, time_lags=[-1,0,1]):
        id_col = self.id_col
        target_col = self.target_col
        stat_cols = self.static_num_col_list + self.static_cat_col_list
        if len(stat_cols)==0:
            # add a dummy column 'dummy_static_col'
            data['dummy_static_col'] = 'dummy'
            stat_cols = ['dummy_static_col']
        temp_num_cols = self.temporal_known_num_col_list + self.temporal_unknown_num_col_list
        temp_cat_cols = self.temporal_known_cat_col_list + self.temporal_unknown_cat_col_list
        
        data = data[[id_col] + [self.time_index_col] + [target_col] + stat_cols + temp_num_cols + temp_cat_cols]
        data = self.sort_dataset(data)
        
        # stationarity test fn.
        def adf_test(x):
            for col in [target_col]+temp_num_cols:
                result = adfuller(x[col].values)
                adf_stat = result[0]
                p_val = result[1]
                critical_val = min(list(result[4].values()))
                if p_val > 0.01:
                    if adf_stat > critical_val:
                        x['{}_stationary'.format(col)] = 0
                    else:
                        # stationary
                        x['{}_stationary'.format(col)] = 1
                else:
                    # inconclusive -- treat as stationary
                    x['{}_stationary'.format(col)] = 1
            
            return x
        
        # pearsons & spearsons r     
        def pearson_coeff(x, lag, id_col):
            # reduce temporal/auto correlation using differencing
            for col in [target_col]+temp_num_cols:
                if x['{}_stationary'.format(col)].max()==0:
                    x[col] = x[col].diff(periods=1)

            x = x.dropna(subset=[target_col]+temp_num_cols)

            x[temp_num_cols] = x[temp_num_cols].shift(periods=lag)
            x = x.dropna(subset=temp_num_cols)

            p_df = x[[target_col]+temp_num_cols].corr(method='pearson')
            p_df = p_df.reset_index()
            p_df.rename(columns={'index':'level_2'}, inplace=True)
            p_df = p_df[p_df['level_2']==target_col]
            p_df.drop(columns=['level_2',target_col], inplace=True)
            for col in temp_num_cols:
                p_df.rename(columns={col:'{}_pearson_coeff'.format(col)}, inplace=True)
                
            s_df = x[[target_col]+temp_num_cols].corr(method='spearman')
            s_df = s_df.reset_index()
            s_df.rename(columns={'index':'level_2'}, inplace=True)
            s_df = s_df[s_df['level_2']==target_col]
            s_df.drop(columns=['level_2',target_col], inplace=True)
            for col in temp_num_cols:
                s_df.rename(columns={col:'{}_spearman_coeff'.format(col)}, inplace=True)

            x = x[[id_col]].drop_duplicates().reset_index(drop=True)
            corr_df = pd.concat([x, p_df, s_df], axis=1)
            corr_df.fillna(method='ffill', inplace=True)
            corr_df['lag'] = lag
            
            return corr_df
        
        # MI specific functions
        rng = np.random.default_rng(1234) 
        
        def mi(x, id_col):
            # reduce temporal/auto correlation using differencing
            for col in [target_col]+temp_num_cols:
                if x['{}_stationary'.format(col)].max()==0:
                    x[col] = x[col].diff(periods=1)
            
            x = x.dropna(subset=[target_col]+temp_num_cols)
            
            # Add random noise to avoid discrete/low resolution distributions. MI goes to -inf otherwise
            for col in temp_num_cols:
                x[col] += rng.normal(0, 1e-6, size=len(x))
            x[target_col] += rng.normal(0, 1e-6, size=len(x))
                
            # lags
            lags = time_lags
            emi = estimate_mi(x[target_col], x[temp_num_cols], lag=lags, normalize=True, preprocess=True)
            emi = emi.reset_index()
            emi.rename(columns={'index':'lag'}, inplace=True)
            for col in temp_num_cols:
                emi.rename(columns={col:'{}_MI_Continuous'.format(col)}, inplace=True)

            x = x[[id_col]].drop_duplicates().reset_index(drop=True)
            emi = pd.concat([x, emi], axis=1)
            emi.fillna(method='ffill', inplace=True)
            
            return emi
       
        def mi_discrete(x, id_col):
            # reduce temporal/auto correlation using differencing
            for col in [target_col]:
                if x['{}_stationary'.format(col)].max()==0:
                    x[col] = x[col].diff(periods=1)

            x = x.dropna(subset=[target_col] + temp_cat_cols)
            
            temp_cat_arr = x[temp_cat_cols].values
            x[target_col] += rng.normal(0, 1e-6, size=len(x))
            target_arr = x[target_col].values
            
            # lags
            lags = time_lags
            emi = estimate_mi(target_arr, temp_cat_arr, discrete_x=True, lag=lags, normalize=False)
            emi = normalize_mi(emi)
            emi_df = pd.DataFrame(emi)
            emi_df.columns = temp_cat_cols
            emi_df['lag'] = pd.Series(lags) 
            
            for col in temp_cat_cols:
                emi_df.rename(columns={col:'{}_MI_Discrete'.format(col)}, inplace=True)
    
            x = x[[id_col]].drop_duplicates().reset_index(drop=True)
            emi_df = pd.concat([x, emi_df], axis=1)
            emi_df.fillna(method='ffill', inplace=True)
            
            return emi_df
        
        # filter out ids with too little data
        data = data.groupby([id_col]).filter(lambda x: len(x) > max(10,(len(time_lags) + 6)))
        data = data.groupby([id_col]).filter(lambda x: len(x[target_col].unique()) > 1)
        
        # obtain metrics for each id
        groups = data.groupby(id_col)
        
        # calculate stationarity columns
        adf_df = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(adf_test)(x) for _,x in groups)
        data = pd.concat(adf_df, axis=0)
        data = data.reset_index(drop=True)
        
        # pearson & spearman corr coeff df
        groups = data.groupby(id_col)
        
        corr_df_list = []
        for lag in time_lags:
            corr = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(pearson_coeff)(x, lag, id_col) for _,x in groups)
            df = pd.concat(corr, axis=0)
            df = df.reset_index(drop=True)
            corr_df_list.append(df)
        
        corr_df = pd.concat(corr_df_list, axis=0)
        corr_df = corr_df.reset_index(drop=True)
        
        # MI df
        groups = data.groupby(id_col)
        
        emi = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(mi)(x, id_col) for _,x in groups)            
        mi_df = pd.concat(emi, axis=0)
        mi_df= mi_df.reset_index(drop=True)
        
        # corr_df, mi merge
        corr_df = corr_df.merge(mi_df, on=[id_col,'lag'], how='left')
        
        # MI Discrete df
        groups = data.groupby(id_col)
        
        emi_disc = Parallel(n_jobs=self.PARALLEL_DATA_JOBS, batch_size=self.PARALLEL_DATA_JOBS_BATCHSIZE)(delayed(mi_discrete)(x, id_col) for _,x in groups)            
        mi_disc_df = pd.concat(emi_disc, axis=0)
        mi_disc_df= mi_disc_df.reset_index(drop=True)
       
        # corr_df, mi disc merge
        corr_df = corr_df.merge(mi_disc_df, on=[id_col,'lag'], how='left')
        
        return corr_df
