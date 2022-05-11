#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as m
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import shutil
import pickle
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import pandas as pd
import random


# Distribution Sampling functions

# negbin
def negbin_sample(mu, alpha, n_samples=1):
    tol = 1e-5
    r = 1.0 / alpha
    theta = alpha * mu
    r = tf.minimum(tf.maximum(tol, r), 1e10)
    theta = tf.minimum(tf.maximum(tol, theta), 1e10)
    beta = 1/theta
    x = tf.minimum(tf.random.gamma(shape=[n_samples], alpha=r, beta=beta), 1e6)
    x = tf.reduce_mean(x, axis=0)
    sample = tf.stop_gradient(tf.random.poisson(shape=[n_samples],lam=x))
    sample = tf.reduce_mean(sample, axis=0)
    return sample

def negbin_multi_sample(mu, alpha, n_samples=1):
    samples = []
    for i in range(n_samples):
        samples.append(negbin_sample(mu, alpha))
    sample_spread = tf.stack(samples, axis=1) #[batch,n_samples,1]
    return tf.reduce_mean(sample_spread, axis=1)

# normal
def normal_sample(mu, std, n_samples=1):
    dist = tfd.Normal(mu,std)
    dist.reparameterization_type == tfd.FULLY_REPARAMETERIZED
    return tf.reduce_mean(dist.sample(sample_shape=n_samples), axis=0)

# student's t sample
def student_sample(df, mu, std, n_samples=1):
    dist = tfd.StudentT(df,mu,std)
    dist.reparameterization_type == tfd.FULLY_REPARAMETERIZED
    return tf.reduce_mean(dist.sample(sample_shape=n_samples), axis=0)

# Poisson
def poisson_sample(mu, n_samples=1):
    dist = tfd.Poisson(rate=mu)
    dist.reparameterization_type == tfd.FULLY_REPARAMETERIZED
    return tf.reduce_mean(dist.sample(sample_shape=n_samples), axis=0)

# Gumbel
def GumbelSample(a, b, n_samples=1):
    dist = tfd.Gumbel(loc=a, scale=b)
    return tf.reduce_mean(tf.stop_gradient(dist.sample(sample_shape=n_samples)), axis=0)

# Building Blocks

# Local Range Convolutional Attention

# Positional Encoding

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
  
@tf.function
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Mask Creation

def create_padding_mask(seq):
    seq = tf.cast(tf.math.less(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
  
# Self-Attention

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

# Conv 1x1 input processing

class Conv1x1(tf.keras.layers.Layer):
    def __init__(self, d_model, N):
        super(Conv1x1, self).__init__()
        
        self.d_model = d_model  # Model Dimensiolnality
        self.N = N              # Number of temporal variables
        
        # 1x1 Conv Layers (1 each for N variables)
        self.conv_layers = [tf.keras.layers.Conv1D(filters=self.d_model, 
                                                   kernel_size=1, 
                                                   strides=1, 
                                                   data_format='channels_last') for _ in range(self.N)]
    def call(self, x, training):
        # x.shape: [(Batch, T, V1_dim), (Batch, T, V2_dim), ..., (Batch, T, V'N'_dim)]
        
        conv_out_list = []
        for i,layer in enumerate(self.conv_layers):
            conv_out_list.append(layer(x[i], training=training))
        
        # stack conv o/ps
        conv_out = tf.stack(conv_out_list, axis=1) # Shape: (Batch,N,T,d_model)
        
        return conv_out

    
# Temporal Conv Self-Attention variable-wise

class LocalRangeConvAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_sizes):
        super( LocalRangeConvAttention, self).__init__()
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        
        # Obtain Q,K,V for multiple kernel sizes
        self.causal_conv_list_of_lists = []
        
        for k in self.kernel_sizes:
            q_layer = tf.keras.layers.Conv1D(filters=self.d_model, 
                                             kernel_size=k, 
                                             strides=1, 
                                             padding='causal', 
                                             data_format='channels_last')
            
            k_layer = tf.keras.layers.Conv1D(filters=self.d_model, 
                                             kernel_size=k, 
                                             strides=1, 
                                             padding='causal', 
                                             data_format='channels_last')
            
            v_layer = tf.keras.layers.Conv1D(filters=self.d_model, 
                                             kernel_size=1,
                                             strides=1, 
                                             padding='causal', 
                                             data_format='channels_last')
            
            self.causal_conv_list_of_lists.append([q_layer, k_layer, v_layer])
        
        self.linear_projection = tf.keras.layers.Dense(self.d_model)
            
        
    def call(self, x, mask, training):
        # x.shape: (Batch, N, T, d_model)
        
        batch_size = tf.shape(x)[0]
        
        attention_out_list = []
        for k, layer_list in enumerate(self.causal_conv_list_of_lists):
            q = layer_list[0](x, training=training) # (Batch, N, T, d_model)
            k = layer_list[1](x, training=training) # (Batch, N, T, d_model)
            v = layer_list[2](x, training=training) # (Batch, N, T, d_model)
                    
            # Apply self attention
            attention_out, attention_weights = scaled_dot_product_attention(q, k, v, mask) # (Batch, N, T, d_model)
                
            # append to list for concatentation later
            attention_out_list.append(attention_out)
            
        # Attention for variable v: (Batch, N, T, d_model*k)
        output = tf.concat(attention_out_list, axis=-1)
            
        # Linear Projection: (Batch, N, T, d_model)
        output = self.linear_projection(output, training=training)
       
        return output


# Spatial Conv Attention

def reverse_shuffle(shuf_idx, order):
    l_out= [0]*len(shuf_idx)
    for i,j in enumerate(shuf_idx):
        l_out[j] = order[i]
    return l_out
    

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask, training):
        batch_size = tf.shape(q)[0]
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask) # (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        output = tf.reshape(scaled_attention, (batch_size, 1, -1, self.num_heads*self.depth))  # (batch_size, 1, seq_len_q, d_model)

        return output
      
        
class GroupRangeConvAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_vars, kernel_size, num_shuffle):
        super( GroupRangeConvAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.k = kernel_size
        self.s = 1
        self.m = num_shuffle
        self.N = num_vars
        if self.N % self.k == 0:
            self.Ng = max(int(self.N/self.k), 1)
        else:
            self.Ng = int(self.N / self.k) + 1
        self.conv_layer_list = [tf.keras.layers.Conv1D(filters=self.d_model, 
                                                       kernel_size=self.k, 
                                                       strides=self.s,
                                                       padding="same", 
                                                       data_format='channels_last') for _ in range(self.Ng)]
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.dense = tf.keras.layers.Dense(self.d_model)
  
    def call(self, x, mask, training):
        # x.shape: (Batch, N, T, d_model)
        
        batch_size= tf.shape(x)[0]
        T = tf.shape(x)[2]
       
        group_attn_list = []
        for i in range(self.m):
            # shuffle x across 'N' dimension
            order =  np.arange(self.N)
            indices = np.arange(self.N)
            np.random.seed(i+1)
            np.random.shuffle(indices)
            reverse_indices = reverse_shuffle(indices, order)
            
            xs = tf.gather(params=x, indices=indices, axis=1)
            
            # group-wise conv attn
            conv_attn_list = []
            for g in range(self.Ng):
                xg = xs[:, self.k*g:self.k*(g+1), :, :]                         # (Batch, Ng, T, d_model)
                xg = tf.transpose(xg, perm=[0, 2, 1, 3])                        # (batch_size, T, Ng, d_model)
                xg = tf.reshape(xg, (batch_size, T, -1))                        # (Batch, T, d_model*Ng)
                xg = self.conv_layer_list[g](xg, training=training)             # (Batch, T, d_model)
                attn = self.mha(xg, xg, xg, mask, training=training)            # (Batch, T, d_model)
                conv_attn_list.append(attn)
                
            conv_attn = tf.concat(conv_attn_list, axis=1)  # (Batch, num_groups, T, d_model)
            
            # Repeat operation
            conv_attn = tf.repeat(conv_attn, repeats=self.k, axis=1) # (Batch, Ng*k, T, d_model)
           
            if self.Ng*self.k > self.N:
                conv_attn = conv_attn[:,:self.N,:,:]                      # (Batch, N, T, d_model)
                
            # Position Align Operation
            xattn = tf.gather(params=conv_attn, indices=reverse_indices, axis=1)  # All variables in order again
            
            # append to group-wise attn list
            group_attn_list.append(xattn)
        
        # Concat & linear project groupwise attn
        global_attn = tf.concat(group_attn_list, axis=-1)     # (Batch, N, T, d_model*m)
        
        # linear projection
        global_attn = self.dense(global_attn)                 # (Batch, N, T, d_model)
        
        return global_attn

                      
# Point-wise FFN

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])

# Temporal Attn Layer

class TemporalAttnLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, kernel_size_list, dropout_rate):
        super(TemporalAttnLayer, self).__init__()

        self.temporal_attn = LocalRangeConvAttention(d_model, kernel_size_list)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, mask, training):

        attn_output = self.temporal_attn(x, mask, training=training)    # (batch_size, N, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)                         # (batch_size, N, input_seq_len, d_model)
    
        ffn_output = self.ffn(out1, training=training)                  # (batch_size, N, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)                       # (batch_size, N, input_seq_len, d_model)
    
        return out
                
                
# Spatial Attn  Layer 

class EncoderSpatialAttnLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, num_vars, kernel_size, num_shuffle, dropout_rate):
        super(EncoderSpatialAttnLayer, self).__init__()

        self.spatial_attn = GroupRangeConvAttention(d_model, num_heads, num_vars, kernel_size, num_shuffle)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, mask, training):

        attn_output = self.spatial_attn(x, mask, training=training)    # (batch_size, N, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)                        # (batch_size, N, input_seq_len, d_model)
    
        ffn_output = self.ffn(out1, training=training)                 # (batch_size, N, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)                       # (batch_size, N, input_seq_len, d_model)
    
        return out
    

# Decoder Spatial Attn  Layer 

class DecoderSpatialAttnLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, num_vars, kernel_size, num_shuffle, dropout_rate):
        super(DecoderSpatialAttnLayer, self).__init__()

        self.self_spatial_attn = GroupRangeConvAttention(d_model, num_heads, num_vars, kernel_size, num_shuffle)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_attn, training):

        attn_output = self.self_spatial_attn(x, mask=None, training=training)    # (batch_size, N, output_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.layernorm1(x + attn_output)  
        
        cross_attn_output, _ = scaled_dot_product_attention(attn_output, enc_attn, enc_attn, mask=None)  # (batch_size, N, output_seq_len, d_model)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        cross_attn_output = self.layernorm2(cross_attn_output + attn_output) 
    
        ffn_output = self.ffn(cross_attn_output, training=training)                 # (batch_size, N, output_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(cross_attn_output + ffn_output)                       
    
        return out # (batch_size, N, input_seq_len, d_model)
                
                
# STCTN Layer

class STCTN(tf.keras.layers.Layer):
    def __init__(self, 
                 num_layers, 
                 d_model, 
                 num_heads, 
                 num_vars,
                 dff, 
                 p_len, 
                 f_len, 
                 temporal_kernel_size_list, 
                 spatial_kernel_size,
                 num_shuffle,
                 dropout_rate):
        
        super(STCTN, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.p_len = p_len
        self.f_len = f_len
        self.seq_len = p_len + f_len
        self.pos_encoding = positional_encoding(self.seq_len, self.d_model)
        
        # Encoder Layers
        self.enc_temporal_attn_layers = [TemporalAttnLayer(d_model, dff, temporal_kernel_size_list, dropout_rate) for _ in range(num_layers)]
        self.enc_spatial_attn_layers = [EncoderSpatialAttnLayer(d_model, dff, num_heads, num_vars, spatial_kernel_size, num_shuffle, dropout_rate) for _ in range(num_layers)]
        self.enc_conv1x1 = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=1, data_format='channels_last')
        
        # Decoder Layers
        self.dec_temporal_attn_layers = [TemporalAttnLayer(d_model, dff, temporal_kernel_size_list, dropout_rate) for _ in range(num_layers)]
        self.dec_spatial_attn_layers = [DecoderSpatialAttnLayer(d_model, dff, num_heads, num_vars, spatial_kernel_size, num_shuffle, dropout_rate) for _ in range(num_layers)]
        self.dec_conv1x1_l1 = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=1, data_format='channels_last')
        
    def call(self, x, mask, training):
        # x.shape: (Batch, N, T, d_model)
        
        enc_spatial_in = x[:,:,:self.p_len,:] 
        
        # CPE
        x += self.pos_encoding[:, tf.newaxis, :self.seq_len, :]
        
        enc_temporal_in = x[:,:,:self.p_len,:] 
        dec_in = x[:,:,-self.f_len:,:] 
        
        for layer in self.enc_temporal_attn_layers:
            enc_temporal_in = layer(enc_temporal_in, mask, training=training)
            
        for layer in self.enc_spatial_attn_layers:
            enc_spatial_in = layer(enc_spatial_in, mask, training=training)
            
        # concat encoder spatial & temporal o/p
        enc_out = tf.concat([enc_spatial_in, enc_temporal_in], axis=-1)
        enc_out = self.enc_conv1x1(enc_out, training=training)
        
        # decoder temporal
        for layer in self.dec_temporal_attn_layers:
            dec_in = layer(dec_in, None, training=training)
            
        # decoder spatial
        for layer in self.dec_spatial_attn_layers:
            dec_in = layer(dec_in, enc_out, training=training)
        
        # decoder o/p
        dec_out = self.dec_conv1x1_l1(dec_in, training=training)
        
        return dec_out  # (batch_size, N, f_len, d_model)
                    

# Model

class STCTN_Model(tf.keras.Model):
    def __init__(self,
                 num_vars,
                 num_layers, 
                 d_model, 
                 num_heads, 
                 dff, 
                 p_len, 
                 f_len, 
                 temporal_kernel_size_list, 
                 spatial_kernel_size,
                 num_shuffle,
                 num_quantiles,
                 loss_fn,
                 dropout_rate):
        
        super(STCTN_Model, self).__init__()
        
        self.p_len = p_len
        self.f_len = f_len
        self.num_vars = num_vars
        self.loss_fn = loss_fn
        self.num_quantiles = num_quantiles
        
        self.conv1x1_linear_transform_layer = Conv1x1(d_model, num_vars)
        
        self.model = STCTN(num_layers, 
                          d_model, 
                          num_heads, 
                          num_vars,
                          dff, 
                          p_len, 
                          f_len, 
                          temporal_kernel_size_list, 
                          spatial_kernel_size, 
                          num_shuffle,
                          dropout_rate)
        
        if self.loss_fn == 'Point':
            self.final_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, data_format='channels_last')
        elif self.loss_fn == 'Poisson':
            self.final_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='softplus', data_format='channels_last')
        elif self.loss_fn == 'Quantile':
            self.quantile_layer = tf.keras.layers.Conv1D(filters=num_quantiles, kernel_size=1, data_format='channels_last')
        elif self.loss_fn == 'Normal':
            self.m_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, data_format='channels_last')
            self.s_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='softplus', data_format='channels_last')
        elif self.loss_fn == 'Negbin':
            self.m_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='softplus', data_format='channels_last')
            self.a_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='softplus', data_format='channels_last')
        else:
            raise ValueError("Invalid Loss Function!")
        
    def call(self, inputs, training):
        # inputs: [[(Batch, T, V1_dim), (Batch, T, V2_dim), ..., (Batch, T, V'N'_dim)], mask, scale]
        # mask, scale : (Batch, p_len, 1)
        
        x, mask, scale = inputs
        model_in = self.conv1x1_linear_transform_layer(x, training=training)
        model_out = self.model(model_in, mask, training=training)            # (batch_size, N, f_len, d_model)
        
        # scale
        scale = scale[:,-1:,:]
        
        # forecasts
        if self.loss_fn == 'Point':
            out = self.final_layer(model_out, training=training)         # (batch_size, N, f_len, 1)
            out = out[:,0,:,:]                                           # (batch_size, f_len, 1). Retaining only target forecast
        
        elif self.loss_fn == 'Poisson':
            mean = self.final_layer(model_out, training=training)        # (batch_size, N, f_len, 1)
            mean = mean[:,0,:,:]*scale                                   # (batch_size, f_len, 1)
            parameters = mean
            out = poisson_sample(mu=mean)
        
        elif self.loss_fn == 'Quantile':                                 
            out = self.quantile_layer(model_out, training=training)      # (batch_size, N, f_len, num_quantiles)
            out = out[:,0,:,:]                                           # (batch_size, f_len, num_quantiles)
        
        elif self.loss_fn == 'Normal':
            mean = self.m_layer(model_out, training=training)             # (batch_size, N, f_len, 1)
            mean = mean[:,0,:,:]*scale                                    # (batch, f_len, 1)
            stddev = self.s_layer(model_out, training=training)           # (batch_size, N, f_len, 1)
            stddev = stddev[:,0,:,:]*scale                                # (batch, f_len, 1)
            parameters = tf.concat([mean, stddev], axis=-1)
            out = normal_sample(mean, stddev)
            
        elif self.loss_fn == 'Negbin':
            mean = self.m_layer(model_out, training=training)                      # (batch_size, N, f_len, 1)
            mean = mean[:,0,:,:]*scale                                             # (batch, f_len, 1)
            alpha = self.a_layer(model_out, training=training)                     # (batch_size, N, f_len, 1)
            alpha = alpha[:,0,:,:]*tf.sqrt(scale)                                  # (batch, f_len, 1)
            parameters = tf.concat([mean, alpha], axis=-1)
            out = negbin_sample(mean, alpha)
            
        else:
            raise ValueError('Invalid Loss Function.')
        
        if self.loss_fn == 'Point' or self.loss_fn == 'Quantile':
            return out, scale
        else:
            return out, parameters

# wrapper class

class SpatioTemporalTransformer(tf.keras.Model):
    def __init__(self,
                 col_index_dict,
                 vocab_dict,
                 num_layers,
                 num_heads,
                 d_model,
                 temporal_kernel_size_list,
                 spatial_kernel_size,
                 num_shuffle,
                 forecast_horizon,
                 max_inp_len,
                 loss_type,
                 num_quantiles,
                 dropout_rate):

        super(SpatioTemporalTransformer, self).__init__()
        
        self.hist_len = int(max_inp_len)
        self.f_len = int(forecast_horizon)
        self.loss_type = loss_type
        self.col_index_dict = col_index_dict
        self.vocab_dict = vocab_dict
        self.temporal_kernel_size_list = temporal_kernel_size_list
        self.spatial_kernel_size = spatial_kernel_size
        self.num_shuffle = num_shuffle
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.max_inp_len = max_inp_len
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        
        # Create static cat embedding layers
        self.stat_col_details = vocab_dict.get('static_cat_indices', None)
        if self.stat_col_details:
            self.stat_lookup_tables = {}
            self.stat_embed_layers = {}
            stat_col_names = self.stat_col_details[0]
            stat_col_vocab = self.stat_col_details[1]
            stat_col_emb_dim = self.stat_col_details[2]
            for i,(colname, values, emb) in enumerate(zip(stat_col_names, stat_col_vocab, stat_col_emb_dim)):
                values = tf.convert_to_tensor(values, dtype=tf.string)
                cat_indices = tf.range(len(values), dtype = tf.int64)
                cat_init = tf.lookup.KeyValueTensorInitializer(values, cat_indices)
                cat_lookup_table = tf.lookup.StaticVocabularyTable(cat_init, 1)
                self.stat_lookup_tables[colname] = cat_lookup_table
                self.stat_embed_layers[colname] = tf.keras.layers.Embedding(input_dim = len(values) + 1, 
                                                                            output_dim = emb,
                                                                            name = "embedding_layer_{}".format(colname))
        # Create temporal known cat embedding layers
        self.temporal_known_col_details = vocab_dict.get('temporal_known_cat_indices', None)
        if self.temporal_known_col_details:
            self.temporal_known_lookup_tables = {}
            self.temporal_known_embed_layers = {}
            temporal_known_col_names = self.temporal_known_col_details[0]
            temporal_known_col_vocab = self.temporal_known_col_details[1]
            temporal_known_col_emb_dim = self.temporal_known_col_details[2]
            for i,(colname, values, emb) in enumerate(zip(temporal_known_col_names, temporal_known_col_vocab, temporal_known_col_emb_dim)):
                values = tf.convert_to_tensor(values, dtype=tf.string)
                cat_indices = tf.range(len(values), dtype = tf.int64)
                cat_init = tf.lookup.KeyValueTensorInitializer(values, cat_indices)
                cat_lookup_table = tf.lookup.StaticVocabularyTable(cat_init, 1)
                self.temporal_known_lookup_tables[colname] = cat_lookup_table
                self.temporal_known_embed_layers[colname] = tf.keras.layers.Embedding(input_dim = len(values) + 1, 
                                                                                      output_dim = emb,
                                                                                      name = "embedding_layer_{}".format(colname))
        # Create temporal unknown cat embedding layers
        self.temporal_unknown_col_details = vocab_dict.get('temporal_unknown_cat_indices', None)
        if self.temporal_unknown_col_details:
            self.temporal_unknown_lookup_tables = {}
            self.temporal_unknown_embed_layers = {}
            temporal_unknown_col_names = self.temporal_unknown_col_details[0]
            temporal_unknown_col_vocab = self.temporal_unknown_col_details[1]
            temporal_unknown_col_emb_dim = self.temporal_unknown_col_details[2]
            for i,(colname, values, emb) in enumerate(zip(temporal_unknown_col_names, temporal_unknown_col_vocab, temporal_unknown_col_emb_dim)):
                values = tf.convert_to_tensor(values, dtype=tf.string)
                cat_indices = tf.range(len(values), dtype = tf.int64)
                cat_init = tf.lookup.KeyValueTensorInitializer(values, cat_indices)
                cat_lookup_table = tf.lookup.StaticVocabularyTable(cat_init, 1)
                self.temporal_unknown_lookup_tables[colname] = cat_lookup_table
                self.temporal_unknown_embed_layers[colname] = tf.keras.layers.Embedding(input_dim = len(values) + 1, 
                                                                                        output_dim = emb,
                                                                                        name = "embedding_layer_{}".format(colname))
        
        # columns names & indices
        self.target_col_name, self.target_index = self.col_index_dict.get('target_index')
        self.stat_num_col_names, self.stat_num_indices = self.col_index_dict.get('static_num_indices')
        self.stat_cat_col_names, self.stat_cat_indices = self.col_index_dict.get('static_cat_indices')
        self.known_num_col_names, self.known_num_indices = self.col_index_dict.get('temporal_known_num_indices')
        self.unknown_num_col_names, self.unknown_num_indices = self.col_index_dict.get('temporal_unknown_num_indices')
        self.known_cat_col_names, self.known_cat_indices = self.col_index_dict.get('temporal_known_cat_indices')
        self.unknown_cat_col_names, self.unknown_cat_indices = self.col_index_dict.get('temporal_unknown_cat_indices')
        
        # num_vars
        self.num_vars = len(self.stat_num_indices) + len(self.stat_cat_indices) + len(self.known_num_indices) +                    len(self.unknown_num_indices) + len(self.known_cat_indices) + len(self.unknown_cat_indices) + 3 
        
        # create model  
        self.model = STCTN_Model(num_vars = self.num_vars,
                                 num_layers = self.num_layers, 
                                 d_model = self.d_model, 
                                 num_heads = self.num_heads, 
                                 dff = int(self.d_model*2), 
                                 p_len = self.hist_len, 
                                 f_len = self.f_len, 
                                 temporal_kernel_size_list = self.temporal_kernel_size_list, 
                                 spatial_kernel_size = self.spatial_kernel_size,
                                 num_shuffle = self.num_shuffle,
                                 num_quantiles = self.num_quantiles,
                                 loss_fn = self.loss_type,
                                 dropout_rate = self.dropout_rate)        
           
    def call(self, inputs, training):
        
        # target
        target = tf.strings.to_number(inputs[:,:,self.target_index:self.target_index+1], out_type=tf.dtypes.float32)
        
        # zero-out future target values
        target = tf.concat([target[:,:self.hist_len,:], tf.math.scalar_mul(0,target[:,-self.f_len:,:])], axis=1)
       
        # ordered col names list
        cols_list = [target]
        
        # known numeric 
        if len(self.known_num_indices)>0:
            for col, i in zip(self.known_num_col_names, self.known_num_indices):
                num_vars = tf.strings.to_number(inputs[:,:,i:i+1], out_type=tf.dtypes.float32)
                cols_list.append(num_vars)
        
        # unknown numeric
        if len(self.unknown_num_indices)>0:
            for col, i in zip(self.unknown_num_col_names, self.unknown_num_indices):
                num_vars = tf.strings.to_number(inputs[:,:,i:i+1], out_type=tf.dtypes.float32)
                # zero-out future values
                num_vars = tf.concat([num_vars[:,:self.hist_len,:], tf.math.scalar_mul(0,num_vars[:,-self.f_len:,:])], axis=1)
                cols_list.append(num_vars)
                
        # known embeddings
        if len(self.known_cat_indices)>0:
            for col, i in zip(self.known_cat_col_names, self.known_cat_indices):
                cat_var = inputs[:,:,i]
                cat_var_id = self.temporal_known_lookup_tables.get(col).lookup(cat_var)
                cat_var_embeddings = self.temporal_known_embed_layers.get(col)(cat_var_id)
                cols_list.append(cat_var_embeddings)
        
        # unknown embeddings
        if len(self.unknown_cat_indices)>0:
            for col, i in zip(self.unknown_cat_col_names, self.unknown_cat_indices):
                cat_var = inputs[:,:,i]
                cat_var_id = self.temporal_unknown_lookup_tables.get(col).lookup(cat_var)
                cat_var_embeddings = self.temporal_unknown_embed_layers.get(col)(cat_var_id)
                # zero-out future values
                cat_var_embeddings = tf.concat([cat_var_embeddings[:,:self.hist_len,:], tf.math.scalar_mul(0, cat_var_embeddings[:,-self.f_len:,:])], axis=1)
                cols_list.append(cat_var_embeddings)
                                     
        # static numeric
        if len(self.stat_num_indices)>0:
            for col, i in zip(self.stat_num_col_names, self.stat_num_indices):
                stat_var = tf.strings.to_number(inputs[:,:,i:i+1], out_type=tf.dtypes.float32)
                cols_list.append(stat_var)
        
        # static embeddings
        if len(self.stat_cat_indices)>0:
            for col, i in zip(self.stat_cat_col_names, self.stat_cat_indices):
                stat_var = inputs[:,:,i]
                stat_var_id = self.stat_lookup_tables.get(col).lookup(stat_var)
                stat_var_embeddings = self.stat_embed_layers.get(col)(stat_var_id)
                cols_list.append(stat_var_embeddings)
                
        # rel_age
        rel_age = tf.strings.to_number(inputs[:,:,-3:-2], out_type=tf.dtypes.float32)
        cols_list.append(rel_age)
        
        # scale
        scale = tf.strings.to_number(inputs[:,:,-2:-1], out_type=tf.dtypes.float32)
        scale_log = tf.math.log(tf.math.sqrt(scale))
        cols_list.append(scale_log)
        
        # mask
        mask = tf.strings.to_number(inputs[:,:,-1:], out_type=tf.dtypes.float32)
        padding_mask = create_padding_mask(mask[:,:self.hist_len,0])
        
        # model process
        o, s = self.model([cols_list, padding_mask, scale], training=training)
         
        return o, s

# train & infer functions 

def SpatioTemporalTransformer_Train(model, 
                                    train_dataset, 
                                    test_dataset, 
                                    loss_type,
                                    loss_function, 
                                    metric, 
                                    learning_rate,
                                    max_epochs, 
                                    min_epochs,
                                    train_steps_per_epoch,
                                    test_steps_per_epoch,
                                    patience,
                                    weighted_training,
                                    model_prefix,
                                    logdir,
                                    opt=None,
                                    clipnorm=None):
    """
     train_dataset, test_dataset: tf.data.Dataset iterator for train & test datasets 
     loss_type: One of ['Point','Quantile','Normal','Poisson','Negbin']
     loss_function: One of the supported loss functions in loss library
     metric: 'MAE' or 'MSE' 
     max_epochs: Max. training epochs
     min_epochs: Min. Training epochs
     *_steps_per_epoch: batches per epoch 
     weighted_training: True/False
     model_prefix: relative or absolute model path with a prefix for a model name
     logdir: tensorflow training logs for tensorboard
        
    """
    @tf.function
    def trainstep(model, optimizer, x_train, y_train, scale, wts, training):
        with tf.GradientTape() as tape:
            o, s = model(x_train, training=training)
            out_len = tf.shape(s)[1]
            if loss_type in ['Normal','Poisson','Negbin']:
                if weighted_training:
                    loss = loss_function(y_train*scale[:,-out_len:,:], [s, wts])
                else:
                    loss = loss_function(y_train*scale[:,-out_len:,:], s)                           
            elif loss_type in ['Point']:
                if weighted_training:                               
                    loss = loss_function(y_train, [o, wts])
                else:
                    loss = loss_function(y_train, o)
            elif loss_type in ['Quantile']:
                if weighted_training:                               
                    loss = loss_function(y_train, [o, wts])
                else:
                    loss = loss_function(y_train, o)                                   
            else:
                raise ValueError("Invalid loss_type specified!")
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_variables) if grad is not None)
        return loss, o
    
    @tf.function
    def teststep(model, x_test, y_test, scale, wts, training):
        o, s = model(x_test, training=training)
        out_len = tf.shape(s)[1]
        if loss_type in ['Normal','Poisson','Negbin']:
            if weighted_training:
                loss = loss_function(y_test*scale[:,-out_len:,:], [s, wts])
            else:
                loss = loss_function(y_test*scale[:,-out_len:,:], s)                           
        elif loss_type in ['Point']:
            if weighted_training:                               
                loss = loss_function(y_test, [o, wts])
            else:
                loss = loss_function(y_test, o)
        elif loss_type in ['Quantile']:
            if weighted_training:                               
                loss = loss_function(y_test, [o, wts])
            else:
                loss = loss_function(y_test, o)                                   
        else:
            raise ValueError("Invalid loss_type specified!")        
        return loss, o
       
    # training specific vars
    if opt is None:
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        optimizer = opt
        optimizer.learning_rate=learning_rate
    
    if clipnorm is None:
        pass
    else:
        optimizer.global_clipnorm = clipnorm
       
    print("lr: ",optimizer.learning_rate.numpy())
    
    # model loss & metric
    train_loss_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss_avg = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    if metric == 'MAE':  
        train_metric = tf.keras.metrics.MeanAbsoluteError('train_mae')
        test_metric = tf.keras.metrics.MeanAbsoluteError('test_mae')
    elif metric == 'MSE':
        train_metric = tf.keras.metrics.MeanSquaredError('train_mse')
        test_metric = tf.keras.metrics.MeanSquaredError('test_mse')
    else:
        raise ValueError("{}: Not a Supported Metric".format(metric))
            
    #logging
    train_log_dir = str(logdir).rstrip('/') +'/train'
    test_log_dir = str(logdir).rstrip('/')+'/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
    # hold results
    train_loss_results = []
    train_metric_results = []
    test_loss_results = []
    test_metric_results = []
        
    # initialize model tracking vars
    
    columns_dict_file = model_prefix + '_col_index_dict.pkl'
    with open(columns_dict_file, 'wb') as f:
        pickle.dump(model.col_index_dict, f)

    vocab_dict_file = model_prefix + '_vocab_dict.pkl'
    with open(vocab_dict_file, 'wb') as f:
        pickle.dump(model.vocab_dict, f)
    
    model_tracker_file = open(model_prefix + '_tracker.txt', mode='w', encoding='utf-8')

    model_tracker_file.write('Spatio Temporal Transformer Training started with following Model Parameters ... \n')
    model_tracker_file.write('----------------------------------------\n')
    model_tracker_file.write('num_layers ' + str(model.num_layers) + '\n')
    model_tracker_file.write('num_heads ' + str(model.num_heads) + '\n')
    model_tracker_file.write('d_model ' + str(model.d_model) + '\n')
    model_tracker_file.write('forecast_horizon ' + str(model.forecast_horizon) + '\n')
    model_tracker_file.write('max_inp_len ' + str(model.max_inp_len) + '\n')
    model_tracker_file.write('dropout_rate ' + str(model.dropout_rate) + '\n')
    model_tracker_file.write('col_index_dict path ' + str(columns_dict_file) + '\n')
    model_tracker_file.write('vocab_dict path ' + str(vocab_dict_file) + '\n')
    model_tracker_file.write('----------------------------------------\n')
    model_tracker_file.write('\n')
    model_tracker_file.flush()
    
    model_list = []
    best_model = None
    time_since_improvement = 0
        
    # train loop
    for epoch in range(max_epochs):
        print("Epoch {}/{}". format(epoch, max_epochs)) 
        for step, (x_batch, y_batch, scale, wts) in enumerate(train_dataset):
            if step > train_steps_per_epoch:
                break
            else:
                train_loss, train_out = trainstep(model, optimizer, x_batch, y_batch, scale, wts, training=True)
                out_len = tf.shape(train_out)[1]
                train_loss_avg.update_state(train_loss)
                if loss_type in ['Normal','Poisson','Negbin']:
                    train_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], train_out)
                elif loss_type in ['Point','Quantile']:
                    train_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], train_out*scale[:,-out_len:,:])
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_avg.result(), step=epoch)
                    tf.summary.scalar('accuracy', train_metric.result(), step=epoch)
          
        for step, (x_batch, y_batch, scale, wts) in enumerate(test_dataset):
            if step > train_steps_per_epoch:
                break
            else:
                test_loss, test_out = teststep(model, x_batch, y_batch, scale, wts, training=False)
                out_len = tf.shape(test_out)[1]
                test_loss_avg.update_state(test_loss)
                if loss_type in ['Normal','Poisson','Negbin']:
                    test_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], test_out)
                elif loss_type in ['Point','Quantile']:
                    test_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], test_out*scale[:,-out_len:,:])
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss_avg.result(), step=epoch)
                    tf.summary.scalar('accuracy', test_metric.result(), step=epoch)
              
        print("Epoch: {}, train_loss: {}, test_loss: {}, train_metric: {}, test_metric: {}".format(epoch, 
                                                                                                    train_loss_avg.result().numpy(),
                                                                                                    test_loss_avg.result().numpy(),
                                                                                                    train_metric.result().numpy(),
                                                                                                    test_metric.result().numpy()))
    
        # record losses & metric in lists
        train_loss_results.append(train_loss_avg.result().numpy())
        train_metric_results.append(train_metric.result().numpy())
        test_loss_results.append(test_loss_avg.result().numpy())
        test_metric_results.append(test_metric.result().numpy())
            
        # reset states
        train_loss_avg.reset_states()
        train_metric.reset_states()
        test_loss_avg.reset_states()
        test_metric.reset_states()
        
        # Save Model
        model_path = model_prefix + '_' + str(epoch) 
        model_list.append(model_path)
        
        # track & save best model
        if test_loss_results[epoch]==np.min(test_loss_results):
            best_model = model_path
            tf.keras.models.save_model(model, model_path)
            # reset time_since_improvement
            time_since_improvement = 0
        else:
            time_since_improvement += 1
            
        model_tracker_file.write('best_model path after epochs ' + str(epoch) + ': ' + best_model + '\n')
        print("Best Model: ", best_model)
            
        # remove older models
        if len(model_list)>patience:
            for m in model_list[:-patience]:
                if m != best_model:
                    try:
                        shutil.rmtree(m)
                    except:
                        pass
                
        if (time_since_improvement > patience) and (epoch > min_epochs):
            print("Terminating Training. Best Model path: {}".format(best_model))
            model_tracker_file.close()
            break
            
        # write after each epoch
        model_tracker_file.flush()
        
    return best_model
        
def SpatioTemporalTransformer_Infer(model, 
                                    inputs, 
                                    loss_type, 
                                    hist_len, 
                                    f_len, 
                                    target_index,
                                    num_quantiles):
    infer_tensor, scale, id_arr, date_arr = inputs
    scale = scale[:,-1:,-1]
    window_len = hist_len + f_len
    output = []
        
    # one-shot forecast
    out, dist = model(infer_tensor, training=False)
    
    if loss_type in ['Normal','Poisson','Negbin']:
        dist = dist.numpy()
        output_arr = dist[:,:,0]
        
    elif loss_type in ['Point']:
        out = out.numpy()
        output_arr = out[:,:,0]
    
    elif loss_type in ['Quantile']:
        out = out.numpy()
        output_arr = out[:,:,:]
                             
    # rescale if necessary
    if loss_type in ['Normal','Poisson','Negbin']:
        output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1), output_arr), axis=1))
        output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value':'forecast'})
        output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
        
    elif loss_type in ['Point']:
        output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1), output_arr*scale.reshape(-1,1)), axis=1))
        output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value':'forecast'})
        output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
    
    elif loss_type in ['Quantile']:
        df_list = []
        for i in range(num_quantiles):
            output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1),output_arr[:,:,i]*scale.reshape(-1,1)), axis=1))
            output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value': f"forecast_{i}"})
            output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
            df_list.append(output_df)
        output_df = pd.concat(df_list, axis=1)
        output_df = output_df.loc[:,~output_df.columns.duplicated()]
        
    # merge date_columns
    date_df = pd.DataFrame(date_arr.reshape(-1,)).rename(columns={0:'period'})
    forecast_df = pd.concat([date_df, output_df], axis=1)
    if loss_type in ['Quantile']:
        for i in range(num_quantiles):
            forecast_df[f"forecast_{i}"] = forecast_df[f"forecast_{i}"].astype(np.float32)
    else:
        forecast_df['forecast'] = forecast_df['forecast'].astype(np.float32)
              
    return forecast_df

# packaging all above in one class

class Spatial_Temporal_Transformer:
    def __init__(self, 
                 col_index_dict,
                 vocab_dict,
                 num_layers,
                 num_heads,
                 d_model,
                 temporal_kernel_size_list,
                 spatial_kernel_size,
                 num_shuffle,
                 forecast_horizon,
                 max_inp_len,
                 loss_type,
                 num_quantiles,
                 dropout_rate=0.1):
        
        self.col_index_dict = col_index_dict
        self.vocab_dict = vocab_dict
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.temporal_kernel_size_list = temporal_kernel_size_list
        self.spatial_kernel_size = spatial_kernel_size
        self.num_shuffle = num_shuffle
        self.forecast_horizon = forecast_horizon
        self.max_inp_len = max_inp_len
        self.loss_type = loss_type
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        self.target_col_name, self.target_index = self.col_index_dict.get('target_index')
    
    def build(self):
        tf.keras.backend.clear_session()
        self.model = SpatioTemporalTransformer(self.col_index_dict,
                                               self.vocab_dict,
                                               self.num_layers,
                                               self.num_heads,
                                               self.d_model,
                                               self.temporal_kernel_size_list,
                                               self.spatial_kernel_size,
                                               self.num_shuffle,
                                               self.forecast_horizon,
                                               self.max_inp_len,
                                               self.loss_type,
                                               self.num_quantiles,
                                               self.dropout_rate)
                
    def train(self, 
              train_dataset, 
              test_dataset,
              loss_function, 
              metric, 
              learning_rate,
              max_epochs, 
              min_epochs,
              train_steps_per_epoch,
              test_steps_per_epoch,
              patience,
              weighted_training,
              model_prefix,
              logdir,
              opt=None,
              clipnorm=None):
        
        # Initialize Weights
        for x,y,s,w in train_dataset.take(1):
            self.model(x, training=False)
        
        best_model = SpatioTemporalTransformer_Train(self.model, 
                                                     train_dataset, 
                                                     test_dataset, 
                                                     self.loss_type,
                                                     loss_function, 
                                                     metric, 
                                                     learning_rate,
                                                     max_epochs, 
                                                     min_epochs,
                                                     train_steps_per_epoch,
                                                     test_steps_per_epoch,
                                                     patience,
                                                     weighted_training,
                                                     model_prefix,
                                                     logdir,
                                                     opt,
                                                     clipnorm)
        return best_model
    
    def load(self, model_path):
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model(model_path)
        
    def infer(self, inputs):
        forecast = SpatioTemporalTransformer_Infer(self.model, inputs, self.loss_type, self.max_inp_len, self.forecast_horizon, self.target_index, self.num_quantiles)    
        return forecast

