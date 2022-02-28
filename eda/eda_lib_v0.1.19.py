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
    def __init__(self, col_dict):
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
        
        def pearson_coeff(x, lag):
            # remove temporal/auto correlation using differencing
            x = x[[target_col]+temp_num_cols].diff(axis=0)
            x.dropna(inplace=True)
            x[temp_num_cols] = x[temp_num_cols].shift(periods=lag)
            x.dropna(inplace=True)
            return x.corr()
            
        # Pearson Correlation Distribution
        plots = []
        for stat_col in stat_cols:
            data = data.groupby([stat_col, id_col]).filter(lambda x: len(x) > (len(time_lags) + 3))
            data = data.groupby([stat_col, id_col]).filter(lambda x: len(x[target_col].unique()) > 1)
            subplots = []
            for lag in time_lags:
                df_corr = data.groupby([stat_col, id_col]).apply(lambda x: pearson_coeff(x, lag)).reset_index()
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
        
        def mi(x):
            # Add random noise to avoid discrete/low resolution distributions. MI goes to -inf otherwise
            for col in temp_num_cols:
                x[col] += rng.normal(0, 1e-6, size=len(x))
            x[target_col] += rng.normal(0, 1e-6, size=len(x))
            
            # Difference series to avoid/reduce autocorrelation. MI reported too high otherwise.
            x = x[[target_col]+temp_num_cols].diff(axis=0)
            x.dropna(inplace=True)
            
            # lags
            lags = time_lags
            
            emi = estimate_mi(x[target_col], x[temp_num_cols], lag=lags, normalize=True, preprocess=True)
            return emi
       
        def mi_discrete(x):
            temp_cat_arr = x[temp_cat_cols].values
            x[target_col] += rng.normal(0, 1e-6, size=len(x))
            target_arr = x[target_col].values
            
            # Difference series to avoid/reduce autocorrelation. MI reported too high otherwise.
            x = x[[target_col]].diff(axis=0)
            x.dropna(inplace=True)
            
            # lags
            lags = time_lags
            
            emi = estimate_mi(target_arr, temp_cat_arr, discrete_x=True, lag=lags, normalize=False)
            emi = normalize_mi(emi)
            emi_df = pd.DataFrame(emi)
            emi_df.columns = temp_cat_cols
            return emi_df
        
        # Non-linear Correlation (MI) Distribution - Numeric Columns
        plots = []  
        for stat_col in stat_cols: 
            data = data.groupby([stat_col, id_col]).filter(lambda x: len(x) > (len(time_lags) + 3))
            df_corr = data.groupby([stat_col, id_col]).apply(lambda x: mi(x)).reset_index()
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
            data = data.groupby([stat_col, id_col]).filter(lambda x: len(x) > (len(time_lags) + 3))
            data = data.groupby([stat_col, id_col]).filter(lambda x: len(x[target_col].unique()) > 1)
            df_corr = data.groupby([stat_col, id_col]).apply(lambda x: mi_discrete(x)).reset_index()
            df_corr['level_2'].replace(to_replace=df_corr['level_2'].unique().tolist(), value=time_lags, inplace=True)
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

