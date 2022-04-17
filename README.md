### A library to easily build & train Transformer models for forecasting.

This library uses the Tensorflow & Tensorflow-Probability deep learning libraries to implement & train the models.  

##### Supported versions:     

Tensorflow [2.4.0+ ]   
Tensorflow-Probability [0.10.0+ ]    

Note: If upgrading Tensorflow, skip v2.6.0 (buggy) & go to 2.7.0 or higher

A typical workflow will look like this:

##### Import basic libraries
````
import tfr
import pandas as pd
import numpy as np
import pprint
````
##### Build the Dataset Object - a uniform interface for creating training, testing & inference datasets
````
# Ensure the dataset meets the following criteria:
a) No NaNs or infs
b) No mixed datatypes in any column
b) No column names may contain spaces

df = pd.read_csv(...)

````
##### Create a dictionary with following column groups based on the dataframe

````
'id_col': Unique identifier for time-series' in the dataset. Mandatory.  
'target_col': Target Column. Mandatory.  
'time_index_col': Any Date or Integer index column that can be used to sort the time-series in ascending order. Mandatory.  
'static_num_col_list': A list of numeric columns which are static features i.e. don't change with time. If N/A specify an empty list: []  
'static_cat_col_list': A list of string/categorical columns which are static features. If N/A specify empty list: []  
'temporal_known_num_col_list': A list of time varying numeric columns which are known at the time of inference for the required Forecast horizon. If N/A spcify empty list [].  
'temporal_unknown_num_col_list': A list of time varying numeric columns for which only historical values are known. If N/A spcify empty list [].  
'temporal_known_cat_col_list': A list of time varying categorical columns which are known at the time of inference for the required Forecast horizon. If N/A spcify empty list [].  
'temporal_unknown_cat_col_list': A list of time varying categorical columns for which only historical values are known. If N/A spcify empty list [].  
'strata_col_list': A list of categorical columns to use for stratified sampling. If N/A specify empty list [].  
'sort_col_list': A list of columns to be used for sorting the dataframe. Typically ['id_col','time_index_col']. Mandatory.  
'wt_col': A numeric column to be used for weighted sampling of time-series'. If N/A specify: None.  

columns_dict = {'id_col':'id',  
                'target_col':'Sales',  
                'time_index_col':'date',  
                'static_num_col_list':[],  
                'static_cat_col_list':['item_id','cat_id','store_id','state_id'],  
                'temporal_known_num_col_list':['abs_age'],  
                'temporal_unknown_num_col_list':['sell_price'],  
                'temporal_known_cat_col_list':['month','wday','Week','event_name_1','event_type_1'],  
                'temporal_unknown_cat_col_list':['snap_CA','snap_TX','snap_WI'],  
                'strata_col_list':['state_id','store_id'],  
                'sort_col_list':['id','date'],  
                'wt_col':'Weight'}  
````                 
##### Create the dataset object using the dictionary defined above.
````
col_dict: Columns grouping dictionary defined above.  
window_len: int(maximum look back history + forecast horizon )    
fh: int(forecast horizon)    
batch: Specifies training & testing batch size. If using stratified sampling, this is the batch size per strata.  
min_nz: Min. no. of non zero values in the Target series within the window_len for it to qualify as a training sample.  
PARALLEL_DATA_JOBS: Option to use parallel processing for training batches generation.  
PARALLEL_DATA_JOBS_BATCHSIZE: Batch size to process within each of the parallel jobs.    
 
data_obj = tfr.tfr_dataset(col_dict=columns_dict,   
                           window_len=26,   
                           fh=13,   
                           batch=16,   
                           min_nz=1,   
                           PARALLEL_DATA_JOBS=1,   
                           PARALLEL_DATA_JOBS_BATCHSIZE=64)                    
````                 
##### Create train & test datasets to be passed to the model (to be built soon).
````
df = Processed Pandas Dataframe read earlier.  
train_till = Date/time_index_col cut-off for training data.   
test_till = Date/time_index_col cut-off for testing data. Typically this will be 'train_till + forecast_horizon'  

trainset, testset = data_obj.train_test_dataset(df,   
                                                train_till=pd.to_datetime('2015-12-31', format='%Y-%M-%d'),   
                                                test_till=pd.to_datetime('2016-01-31', format='%Y-%M-%d'))  
````
##### Obtain Column info dictionary & Vocab dictionary (required arguments for model)  
````
col_index_dict = data_obj.col_index_dict  
vocab = data_obj.vocab_list(df)  
````
##### Create Inference dataset for final predctions. This can be done separately from above.
````
infer_dataset, actuals_df = data_obj.infer_dataset(df,   
                                                   history_till=pd.to_datetime('2015-12-31', format='%Y-%M-%d'),   
                                                   future_till=pd.to_datetime('2016-01-31', format='%Y-%M-%d'))  

where, actuals_df is a dataframe of ground_truths (to be used for evaluation)

````
##### Build Model
````
num_layers: Int. Specify no. of attention layers in the Transformer model. Typical range [1-4]    
num_heads: Int. No. of heads to be used for self attention computation. Typical range [1-4]  
d_model: Int. Model Dimension. Typical range [32,64,128]. Multiple of num_heads.  
forecast_horizon: same as 'fh' defined above.  
max_inp_len: = int(window_len - fh)  
loss_type: One of ['Point','Quantile'] for Point forecasts or ['Normal','Poisson','Negbin'] for distribution based forecasts  
dropout_rate: % Dropout for regularization  
trainset, testset: tf.data.Dataset datasources obtained above  
Returns the model object  

Select a loss_type & loss_function from the following:
   
pprint.pprint(tfr.supported_losses) 

{'Huber': ['loss_type: Point', 'Usage: Huber(delta=1.0, sample_weights=False)'],
 'Negbin': ['loss_type: Negbin', 'Usage: Negbin_NLL_Loss(sample_weights=False)'],
 'Normal': ['loss_type: Normal', 'Usage: Normal_NLL_Loss(sample_weights=False)'],
 'Poisson': ['loss_type: Poisson', 'Usage: Poisson_NLL_Loss(sample_weights=False)'],
 'Quantile': ['loss_type: Quantile', 'Usage: QuantileLoss_v2(quantiles=[0.5], sample_weights=False)'],
 'RMSE': ['loss_type: Point', 'Usage: RMSE(sample_weights=False)']
 }

e.g.
loss_type = 'Quantile' 
loss_fn = QuantileLoss_Weighted(quantiles=[0.6])
  
try:
    del model
except:
    pass
    
model = Simple_Transformer(col_index_dict = col_index_dict,
                           vocab_dict = vocab,
                           num_layers = 2,
                           num_heads = 4,
                           d_model = 64,
                           forecast_horizon = 13,
                           max_inp_len = 13,
                           loss_type = 'Quantile,
                           dropout_rate=0.1)

model.build() 
````                                  
##### Train model  
````
train_dataset, test_dataset: tf.data.Dataset objects  
loss_function: One of the supported loss functions. See the output of pprint.pprint(supported_losses) for usage.  
metric: 'MAE' or 'MSE'  
learning_Rate: Typical range [0.001 - 0.00001]  
max_epochs, min_epochs: Max & min training epochs  
steps_per_epoch: no. of training batches/gradient descent steps per epoch  
patience: how many epochs to wait before terminating in case of non-decreasing loss  
weighted_training: True/False.   
model_prefix: Path where to save models  
logdir: Training logs location. Can be viewed with Tensorboard.  

best_model = model.train(train_dataset=trainset,   
                         test_dataset=testset,
                         loss_function=loss_fn,              
                         metric='MSE',
                         learning_rate=0.0001,
                         max_epochs=2,
                         min_epochs=1,
                         train_steps_per_epoch=10,
                         test_steps_per_epoch=5,
                         patience=2,
                         weighted_training=True,
                         model_prefix='test_models\tfr_model',
                         logdir='test_logs')                         
                         
````
##### Load Model & Predict
Skip 'model.build()' if doing only inference using a saved model.

````
model.load(model_path='test_models\tfr_model_1')
forecast_df = model.infer(infer_dataset)
                     
````  

##### Additionally, you may use feature weighted transformer

````
model = Feature_Weighted_Transformer(col_index_dict = col_index_dict,
                                     vocab_dict = vocab,
                                     num_layers = 2,
                                     num_heads = 4,
                                     d_model = 64,
                                     forecast_horizon = 13,
                                     max_inp_len = 13,
                                     loss_type = 'Quantile,
                                     dropout_rate=0.1)
model.build()

model.train(...) -- usage identical to Simple_Transformer

# Inference returns two outputs:

forecast_df, feature_imp = model.infer(...)

where, 
    forecast_df - forecasts dataframe
    feature_imp - a list of variable importance dataframes in the following order: static_vars_imp_df, historical_vars_imp_df, future_vars_imp_df 

````
##### Baseline Forecasts
````
Prepare the baseline dataset:

baseline_infer_dataset = data_obj.baseline_infer_dataset(df, 
                                                         history_till=pd.to_datetime('2016-01-18', format='%Y-%M-%d'), 
                                                         future_till=pd.to_datetime('2016-01-31', format='%Y-%M-%d'),
                                                         ignore_cols=['event_name_1','event_type_1'])

where, ignore_cols is a list of features to zero out while forecasting so as to eliminate their contribution to total forecast.

Call infer as usual:

baseline_forecast_df, _ = model.infer(baseline_infer_dataset)

````

##### Evaluate Forecasts
````
Evaluation produces two metrics: Forecast_Accuracy & Forecast_Bias expressed as percentages

eval_df = model.evaluate(forecasts=forecast_df, actuals=actuals_df, aggregate_on=['item_id','state_id'])

where, aggregate_on is a list of static categorical columns which provides the level at which to summarize forecast accuracy & bias.
  
````
### New in v0.1.10 - Sparse Attention Transformers
````
Build Model: 

model = Sparse_Simple_Transformer(col_index_dict = col_index_dict,
                                  vocab_dict = vocab,
                                  num_layers = 2,
                                  num_heads = 4,
                                  num_blocks = 2,
                                  kernel_size = 5,  
                                  d_model = 64,
                                  forecast_horizon = 13,
                                  max_inp_len = 14,
                                  loss_type = 'Point',
                                  dropout_rate=0.1)

or 

model = Sparse_Feature_Weighted_Transformer(col_index_dict = col_index_dict,
                                            vocab_dict = vocab,
                                            num_layers = 2,
                                            num_heads = 4,
                                            num_blocks = 2,
                                            kernel_size = 5,
                                            d_model = 64,
                                            forecast_horizon = 13,
                                            max_inp_len = 14,
                                            loss_type = 'Point',
                                            dropout_rate=0.1)

model.build()

Where,
    num_blocks - local attention window size. max_inp_len should be a multiple of num_blocks. 
                 Specify num_blocks > 1 only if working with long sequences. 
    kernel_size - Conv1D causal convolution layer's kernel size. Basically, the look_back_window at each timestep.
                  Typical values: [3,5,7,9]

Train: Same as Feature_Weighted_Transformer

````
### New in v0.1.15
````
Added switch 'low_memory' & 'use_memmap' to the tfr_dataset.train_test_dataset method.
Default: low_memory = True (uses tf.data.Dataset.from_generator API for generating train/test batches). Uses less memory at the expense of speed.
         low_memory = False, uses numpy arrays in tf.data.Dataset.from_tensor_slices(). Initial trainset/testset creation takes time but the training speed improves by 4x.
Default: use_memmap = True (uses numpy.memmap files to reduce memory usage). If False, builds train/test arrays in memory (high mem usage) 

trainset, testset = data_obj.train_test_dataset(df, 
                                               train_till=pd.to_datetime('2015-12-31', format='%Y-%M-%d'), 
                                               test_till=pd.to_datetime('2016-01-31', format='%Y-%M-%d'),
                                               low_memory=False,
                                               use_memmap=False)

````
 ### Added TS Visualization & fixed charset handling to 'utf-8'
````
Plot sample raw time-series:

data_obj.show_ts_samples(data=df, sample_ids=[], n_samples=10, n_col=2, plot_size=(300,600), save=True, filename='ts_samples.html')

Plot sample processed time-series:

data_obj.show_processed_ts_samples(data=df, n_samples=10, n_col=2, plot_size=(300,400), save=True, filename='ts_processed_samples.html')

````

### New in 0.1.18 - EDA package
````
Create Interactive EDA Report

import eda

eda_object = eda.eda(col_dict=columns_dict, PARALLEL_DATA_JOBS=4, PARALLEL_DATA_JOBS_BATCHSIZE=128)  # 'columns_dict' -- similar to the one used in 'tfr_dataset'
eda_object.create_report(data=df, filename='eda_report.html') # df is the pandas dataframe, filename is the full path of the to-be generated report

The create_report method takes a few more arguments:

n_col (default (int): 2) # Configures the grid layout 
plot_size (default (tuple of ints): (400,800)) # (Height,Width) of the plot in pixels
time_lags (default (list of ints): [-1,0,1]) # Used for non-linear correlation density plots between target_col & various numeric & categorical columns for specified lags.
max_static_col_levels (default (int): 100) # If there are too many levels to a static feature, the report can get crowded with redundant plots. This parameter helps skip crowded plots with little utility.

````

### New in 0.1.24 - TFT & Decoder Lags
````
TFT sample usage:

import tft

... tft.tft_dataset
... tft.supported_losses

model = tft.Temporal_Fusion_Transformer(col_index_dict = col_index_dict,
                                    vocab_dict = vocab,
                                    num_layers = 1,
                                    num_heads = 1,
                                    d_model = 32,
                                    forecast_horizon = 13,
                                    max_inp_len = 13,
                                    loss_type = 'Quantile',
                                    num_quantiles=2,
                                    dropout_rate=0.1)

model.build()

Train & Infer methods are identical to other transformers.

For other transformers, one can supply optional parameter decoder_lags (int) during model creation to customize no. of previous target values 
to be used for decoding purpose. Minimum decoder_lags = 1, maximum decoder_lags = "length of the encoder series". 
Default: max(int(len(encoder_timesteps)/4),2) 

````

