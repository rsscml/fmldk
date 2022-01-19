### A library to easily build & train Transformer models for forecasting.

This library uses the Tensorflow & Tensorflow-Probability deep learning libraries to implement & train the models.  

##### Supported versions:     

Tensorflow [2.4.0 - 2.7.0]   
Tensorflow-Probability [0.10.0 - 0.12.0]  

A typical workflow will have these stages:

##### Import basic libraries
````
import tfr
import pandas as pd
import numpy as np
import pprint
````
##### Build the Dataset Object - a uniform interface for creating training, testing & inference datasets
````
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
infer_dataset = data_obj.infer_dataset(df,   
                                       history_till=pd.to_datetime('2015-12-31', format='%Y-%M-%d'),   
                                       future_till=pd.to_datetime('2016-01-31', format='%Y-%M-%d'))  
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
  
model = tfr.Transformer_Model_Builder(col_index_dict = col_index_dict,  
                                      vocab_dict = vocab,  
                                      num_layers = 2,  
                                      num_heads = 4,  
                                      d_model = 64,  
                                      forecast_horizon = 13,  
                                      max_inp_len = 13,  
                                      loss_type = 'Point',  
                                      dropout_rate=0.1,  
                                      trainset = trainset,  
                                      testset = testset)  
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

pprint.pprint(tfr.supported_losses)  

best_model = model.train(train_dataset = trainset,   
                         test_dataset = testset,   
                         loss_function = loss_fn,                
                         metric='MSE',  
                         learning_rate=0.0001,  
                         max_epochs=2,  
                         min_epochs=1,  
                         steps_per_epoch=5,  
                         patience=2,   
                         weighted_training=True,  
                         model_prefix='test_models\wal_test_huber',  
                         logdir='test_logs')  
````
##### Load Model & Predict
````
model = tfr.Load_Model(model_path)
forecast_df = model.infer(infer_dataset)                     
````  
