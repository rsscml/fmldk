#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import gc


# In[ ]:


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

def min_power_of_2(x):
    return m.ceil(m.log2(x))


# In[ ]:


# Model Class - Dense Transformer w/ Variable Selection

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
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# Temporal Conv Self-Attention

class MultiHeadConvAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, kernel_sizes):
        
        super(MultiHeadConvAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
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
                                             data_format='channels_last')
            
            self.causal_conv_list_of_lists.append([q_layer, k_layer, v_layer])
        
        self.linear_projection = tf.keras.layers.Dense(self.d_model)
    
    def split_heads(self, x, batch_size):
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
            
    def call(self, value, key, query, mask, training):
        
        batch_size = tf.shape(query)[0]
        
        attention_out_list = []
        attention_weights_list = []
        for k, layer_list in enumerate(self.causal_conv_list_of_lists):
            
            # apply conv layers
            q = layer_list[0](query, training=training) # (Batch, seq_len_q, d_model)
            k = layer_list[1](key, training=training) # (Batch, seq_len_k, d_model)
            v = layer_list[2](value, training=training) # (Batch, seq_len_v, d_model)
            
            # split heads
            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
            # Apply self attention
            scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask) # (batch_size, num_heads, seq_len_q, depth)
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

            # append to list for concatentation later
            attention_out_list.append(concat_attention)
            attention_weights_list.append(attention_weights)
            
        # Attention o/p for variable v
        output = tf.stack(attention_out_list, axis=1)                # (Batch, k, seq_len_q, d_model)
        output = tf.math.reduce_mean(output, axis=1, keepdims=False) # (Batch, seq_len_q, d_model)
        
        # mean of attention weights
        attention_weights = tf.stack(attention_weights_list, axis=1) # (batch_size, k, num_heads, seq_len_q, depth)
        attention_weights = tf.math.reduce_mean(attention_weights, axis=1, keepdims=False) # (batch_size, num_heads, seq_len_q, depth)
            
        # Linear Projection: (Batch, T, d_model)
        output = self.linear_projection(output, training=training)
       
        return output, attention_weights
    

# Multi-Head Self-Attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask, training):
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q, training=training)  # (batch_size, seq_len, d_model)
        k = self.wk(k, training=training)  # (batch_size, seq_len, d_model)
        v = self.wv(v, training=training)  # (batch_size, seq_len, d_model)
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention, training=training)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights
      
# Point-wise FFN

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                                tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
                               ])
    

# Decoder Layer - Conv Attention

class ConvDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, kernel_sizes, dff, rate):
        super(ConvDecoderLayer, self).__init__()

        self.mha1 = MultiHeadConvAttention(d_model, num_heads, kernel_sizes)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
 
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
       
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, look_ahead_mask, padding_mask, training):
        
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, training=training)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
    
        ffn_output = self.ffn(out1, training=training)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
    
        return out2, attn_weights_block1

# Conv Decoder Module

class ConvDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, kernel_sizes, dff, rate):
        super(ConvDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.dec_layers = [ConvDecoderLayer(d_model, num_heads, kernel_sizes, dff, rate) for _ in range(num_layers)]
        
    def call(self, x, look_ahead_mask, padding_mask, training):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](x, look_ahead_mask, padding_mask, training=training)
      
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        
        return x, attention_weights
    
    
# GRN & Gating

class linear_layer(tf.keras.layers.Layer):
    def __init__(self, size, activation=None, use_time_distributed=False, use_bias=True):
        super(linear_layer, self).__init__()
        self.linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
        if use_time_distributed:
            self.linear = tf.keras.layers.TimeDistributed(self.linear)
    
    def call(self,inputs, training):
        return self.linear(inputs, training=training)
    

class apply_gating_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate=None, use_time_distributed=True, activation=None):
        super(apply_gating_layer, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        if use_time_distributed:
            self.activation_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation=activation))
            self.gated_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))
        else:
            self.activation_layer = tf.keras.layers.Dense(hidden_layer_size, activation=activation)
            self.gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid')
    
    def call(self, inputs, training):
        x = self.dropout(inputs, training=training)
        a = self.activation_layer(x, training=training)
        g = self.gated_layer(x, training=training)
        return tf.keras.layers.Multiply()([a, g]), g
           
def add_and_norm(x_list):
    tmp = tf.keras.layers.Add()(x_list)
    tmp = tf.keras.layers.LayerNormalization()(tmp)
    return tmp

class add_norm_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(add_norm_layer, self).__init__()
        self.add_layer = tf.keras.layers.Add()
        self.norm_layer = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs, training):
        x = self.add_layer(inputs)
        x = self.norm_layer(x)
        return x
    
class gated_residual_network(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, output_size=None, dropout_rate=None, use_time_distributed=True, 
                 additional_context=False, return_gate=False):
        super(gated_residual_network, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        if output_size is None:
            self.output_size = hidden_layer_size
        else: 
            self.output_size = output_size
        if use_time_distributed:
            self.linear = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_size))
        else:
            self.linear = tf.keras.layers.Dense(self.output_size)
        self.additional_context = additional_context
        self.return_gate = return_gate
        self.hidden_1 = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed)
        if additional_context:
            self.context = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed, use_bias=False)
        self.hidden_2 = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed)
        self.gate = apply_gating_layer(self.output_size, dropout_rate=dropout_rate, use_time_distributed=use_time_distributed, activation=None)
        self.add_norm = add_norm_layer()
    
    def call(self, inputs, training):
        
        if self.additional_context:
            x,c = inputs
            skip = self.linear(x, training=training)
            hidden = self.hidden_1(x, training=training)
            hidden = hidden + self.context(c, training=training)
            hidden = tf.keras.layers.Activation('elu')(hidden)
            hidden = self.hidden_2(hidden, training=training)
        else:
            x = inputs
            skip = self.linear(x, training=training)
            hidden = self.hidden_1(x, training=training)
            hidden = tf.keras.layers.Activation('elu')(hidden)
            hidden = self.hidden_2(hidden, training=training)
        
        gating_layer, gate = self.gate(hidden, training=training)
        if self.return_gate:
            grn_out, g = self.add_norm([skip, gating_layer], training), gate
            return grn_out, g
        else:
            grn_out = self.add_norm([skip, gating_layer], training)
            return grn_out
    
    
# Variable selection networks

class static_variable_selection_layer(tf.keras.layers.Layer):
    """
    Takes inputs as a list of embedded/linear transformed tensors
    """
    def __init__(self, hidden_layer_size, output_size, dropout_rate):
        super(static_variable_selection_layer, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
      
    def build(self, input_shape):
        self.num_static = len(input_shape) 
        self.grn_var = [gated_residual_network(self.hidden_layer_size, 
                                               self.output_size, 
                                               self.dropout_rate, 
                                               use_time_distributed=False, 
                                               additional_context=False, 
                                               return_gate=False) for _ in range(self.num_static)]
        
        self.grn_flat = gated_residual_network(self.hidden_layer_size, 
                                               self.num_static, 
                                               self.dropout_rate, 
                                               use_time_distributed=False, 
                                               additional_context=False, 
                                               return_gate=False)
        
    def call(self, inputs, training):
        flatten = tf.concat(inputs, axis=1) #[batch, sum_of_var_dims]
        mlp_outputs = self.grn_flat(flatten, training=training)
        
        static_weights = tf.keras.layers.Activation('softmax')(mlp_outputs) #[batch,num_static]
        weights = tf.expand_dims(static_weights, axis=-1) #[batch,num_static,1]
        
        trans_emb_list = []
        for i in range(self.num_static):
            e = self.grn_var[i](inputs[i], training=training)
            trans_emb_list.append(e)
        
        trans_embedding = tf.stack(trans_emb_list, axis=1) #[batch,num_static,hidden_layer_size]
        
        combined = tf.keras.layers.Multiply()([weights, trans_embedding])
        static_vec = tf.reduce_sum(combined, axis=1)
        
        return static_vec, static_weights

    
class static_contexts(tf.keras.layers.Layer):
    """
    Takes static_vec as input
    """
    def __init__(self, hidden_layer_size, output_size, dropout_rate):
        super(static_contexts, self).__init__()
        self.static_context_variable_selection = gated_residual_network(hidden_layer_size, 
                                                                        output_size, 
                                                                        dropout_rate, 
                                                                        use_time_distributed=False, 
                                                                        additional_context=False, 
                                                                        return_gate=False)
    def call(self, inputs, training):
        static_var_select_vec = self.static_context_variable_selection(inputs, training=training)
        return static_var_select_vec
      
class static_enrichment_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, context, dropout_rate):
        super(static_enrichment_layer, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.context = context
        self.grn_enrich = gated_residual_network(hidden_layer_size=self.hidden_layer_size,
                                                 output_size=None,
                                                 dropout_rate=self.dropout_rate,
                                                 use_time_distributed=True,
                                                 additional_context=self.context,
                                                 return_gate=False)
    def call(self, inputs, training):
        if self.context:
            x,c = inputs
            c = tf.expand_dims(c, axis=1)
            enriched = self.grn_enrich([x,c], training=training)
        else:
            x = inputs
            enriched = self.grn_enrich(x, training=training)
            
        return enriched

# LSTM Layer

class lstm_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, rnn_layers, dropout_rate):
        super(lstm_layer, self).__init__()

        self.rnn_layers = rnn_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.temporal_layer = tf.keras.layers.LSTM(units=hidden_layer_size, return_state=True, return_sequences=True)
        self.gate = apply_gating_layer(hidden_layer_size=self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=True,activation=None)
        self.add_norm = add_norm_layer()

    def call(self, inputs, training):
        decoder_in, init_states = inputs
        lstm_out, enc_h, enc_c = self.temporal_layer(decoder_in, initial_state=init_states, training=training)
        # Apply gating layer
        lstm_out, _ = self.gate(lstm_out, training=training)
        temporal_features = self.add_norm([lstm_out, decoder_in], training=training)
        return temporal_features

      
class lstm_init_states(tf.keras.layers.Layer):
    """
    Takes static_vec as input
    """
    def __init__(self, hidden_layer_size, output_size, dropout_rate):
        super(lstm_init_states, self).__init__()
        
        self.init_h = gated_residual_network(hidden_layer_size, 
                                             output_size, 
                                             dropout_rate, 
                                             use_time_distributed=False, 
                                             additional_context=False,                            
                                             return_gate=False)
        self.init_c = gated_residual_network(hidden_layer_size, 
                                             output_size, 
                                             dropout_rate, 
                                             use_time_distributed=False, 
                                             additional_context=False,                            
                                             return_gate=False)
    def call(self, inputs, training):
        init_h = self.init_h(inputs, training=training)
        init_c = self.init_c(inputs, training=training)
        return init_h, init_c
    

class temporal_variable_selection_layer(tf.keras.layers.Layer):
    """
    Takes inputs as a list of list of embedded/linear transformed tensors of shape [batch,time_steps,emb_dim] & context vec
    """
    def __init__(self, hidden_layer_size, output_size, context, dropout_rate):
        super(temporal_variable_selection_layer, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.context = context
        self.dropout_rate = dropout_rate
  
    def build(self, input_shape):
        if self.context:
            self.num_vars = len(input_shape[0]) 
        else:
            self.num_vars = len(input_shape)
          
        self.grn_var = [gated_residual_network(self.hidden_layer_size, 
                                               self.output_size, 
                                               self.dropout_rate, 
                                               use_time_distributed=True, 
                                               additional_context=False, 
                                               return_gate=False) for _ in range(self.num_vars)]
        
        self.grn_flat = gated_residual_network(self.hidden_layer_size, 
                                               self.num_vars, 
                                               self.dropout_rate, 
                                               use_time_distributed=True, 
                                               additional_context=self.context, 
                                               return_gate=False)
        
    def call(self, x, training):
        if self.context:
            inputs, context = x
            context = tf.expand_dims(context, axis=1)
            flatten = tf.concat(inputs, axis=-1) #[batch,time_steps,sum_of_var_dims]
            mlp_outputs = self.grn_flat([flatten,context], training=training) #[batch,time_steps,num_vars]
        else:
            inputs = x
            flatten = tf.concat(inputs, axis=-1) #[batch,time_steps,sum_of_var_dims]
            mlp_outputs = self.grn_flat(flatten, training=training) #[batch,time_steps,num_vars]
        
        dynamic_weights = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('softmax'))(mlp_outputs) #[batch,time_steps,num_vars]
        weights = tf.expand_dims(dynamic_weights, axis=-1) #[batch,time_steps,num_vars,1]
        
        trans_emb_list = []
        for i in range(self.num_vars):
            e = self.grn_var[i](inputs[i], training=training)
            trans_emb_list.append(e)
        
        trans_embedding = tf.stack(trans_emb_list, axis=2) #[batch,time_steps,num_vars,hidden_layer_size]
        combined = tf.keras.layers.Multiply()([weights, trans_embedding])
        tfr_input = tf.reduce_sum(combined, axis=2) #[batch,time_steps,hidden_layers_size]
       
        return tfr_input, dynamic_weights
      
class all_variable_select_concat_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, output_size, dropout_rate):
        super(all_variable_select_concat_layer, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
  
    def build(self, input_shape):
        self.num_vars = len(input_shape)  
        self.grn_var = [gated_residual_network(self.hidden_layer_size, 
                                               self.output_size, 
                                               self.dropout_rate, 
                                               use_time_distributed=True, 
                                               additional_context=False, 
                                               return_gate=False) for _ in range(self.num_vars)]
        self.grn_flat = gated_residual_network(self.hidden_layer_size, 
                                               self.num_vars, 
                                               self.dropout_rate, 
                                               use_time_distributed=True, 
                                               additional_context=False, 
                                               return_gate=False)
    
        
    def call(self, inputs, training):
        flatten = tf.concat(inputs, axis=-1) #[batch,time_steps,sum_of_var_dims]
        mlp_outputs = self.grn_flat(flatten, training=training) #[batch,time_steps,num_vars]
        dynamic_weights = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('softmax'))(mlp_outputs) #[batch,time_steps,num_vars]
        weights = tf.expand_dims(dynamic_weights, axis=-1) #[batch,time_steps,num_vars,1]
        
        trans_emb_list = []
        for i in range(self.num_vars):
            e = self.grn_var[i](inputs[i], training=training)
            trans_emb_list.append(e)
        
        trans_embedding = tf.stack(trans_emb_list, axis=2) #[batch,time_steps,num_vars,hidden_layer_size]
        
        combined = tf.keras.layers.Multiply()([weights, trans_embedding]) #[batch,time_steps,num_vars,hidden_layer_size]
        batch_size, timesteps = tf.shape(combined)[0], tf.shape(combined)[1]
        tfr_input = tf.reshape(combined, [batch_size, timesteps, -1]) #[batch,time_steps,hidden_layers_size]
        
        return tfr_input, dynamic_weights   
    
    
class final_gating_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate):
        super(final_gating_layer, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.gate = apply_gating_layer(hidden_layer_size=self.hidden_layer_size,
                                       dropout_rate=self.dropout_rate,
                                       use_time_distributed=True,
                                       activation=None)
        self.add_norm = add_norm_layer()

    def call(self, inputs, training):

        attn_out, temporal_features = inputs
        # final gating layer
        attn_out, _ = self.gate(attn_out, training=training)
        # final add & norm
        out = self.add_norm([attn_out, temporal_features], training=training)
        return out


class CausalConvEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers):
        super(CausalConvEncoder, self).__init__()
        self.layers = []
        for i in range(num_layers):
            conv_layer = tf.keras.layers.Conv1D(filters=d_model, kernel_size=2, padding='causal', dilation_rate=2**i)
            self.layers.append(conv_layer)
          
    def call(self, inputs, training):
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        state = x #[:,-1:,:] # return last hidden state 
        return state
    
    
class CausalConvResidualLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, dropout_rate):
        super(CausalConvResidualLayer, self).__init__()
        self.seq_len = seq_len
        num_causal_layers = int(min_power_of_2(self.seq_len))
        
        
        self.conv1x1 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1) 
        self.causalconvlayer1 = CausalConvEncoder(d_model=d_model, num_layers=num_causal_layers)
        self.conv_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.conv_layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv_activation1 = tf.keras.layers.Activation('selu')
        self.conv_add = tf.keras.layers.Add()
        
    def call(self, inputs, training):
        x = inputs
        
        y = self.conv1x1(x)
        x = self.causalconvlayer1(x)
        x = self.conv_layernorm1(x)
        x = self.conv_activation1(x)
        x = self.conv_dropout1(x)
        
        out = self.conv_add([x,y])
        
        return out
        

# Variable Weighted Transformer Model

class SageTransformer(tf.keras.Model):
    def __init__(self,
                 static_variables,
                 num_layers,
                 d_model,
                 num_heads,
                 kernel_sizes,
                 dff,
                 hist_len, 
                 f_len,
                 loss_fn,
                 num_quantiles,
                 rate=0.1):
        super(SageTransformer, self).__init__()
        
        self.hist_len = hist_len
        self.f_len = f_len
        self.loss_fn = loss_fn
        self.stat_variables = static_variables
        self.num_quantiles = num_quantiles
        self.seq_len = int(hist_len + f_len)
        self.num_causal_layers = int(min_power_of_2(self.seq_len)) 
        
        if self.stat_variables:
            self.static_input_layer = static_variable_selection_layer(hidden_layer_size=d_model, output_size=None, dropout_rate=rate)
            self.static_context_layer = static_contexts(hidden_layer_size=d_model, output_size=None, dropout_rate=rate)
            self.enrich_vector_layer = static_contexts(hidden_layer_size=d_model, output_size=None, dropout_rate=rate)
            self.init_states_layer = lstm_init_states(hidden_layer_size=d_model, output_size=d_model, dropout_rate=rate)
        
        self.static_enrich_layer = static_enrichment_layer(hidden_layer_size=d_model, context=self.stat_variables, dropout_rate=rate)
        self.decoder_input_layer = temporal_variable_selection_layer(hidden_layer_size=d_model, output_size=None, context=self.stat_variables, dropout_rate=rate)
        self.recurrent_layer = lstm_layer(hidden_layer_size=d_model, rnn_layers=1, dropout_rate=rate)
        self.decoder = ConvDecoder(num_layers, d_model, num_heads, kernel_sizes, dff, rate)
        self.pff_layer = final_gating_layer(hidden_layer_size=d_model, dropout_rate=rate)
        
        if self.loss_fn == 'Point':
            self.final_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1))
        elif self.loss_fn == 'Poisson':
            self.final_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
        elif self.loss_fn == 'Quantile':
            self.proj_intrcpt = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation=None))
            if self.num_quantiles > 1:
                self.proj_incrmnt = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.num_quantiles-1, activation='softplus'))
        elif self.loss_fn == 'Normal':
            self.m_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1))
            self.s_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
        elif self.loss_fn == 'Negbin':
            self.m_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
            self.a_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
        else:
            raise ValueError("Invalid Loss Function!")
        
    def call(self, inputs, training):
        '''
        shapes:
        static_vars_list: [TensorShape(batch,stat_var1_dim),TensorShape(batch,stat_var2_dim),...]
        decoder_vars_list: [TensorShape(batch, hist+fh_timesteps, dec_var1_dim),...]
        mask: TensorShape(batch,hist+fh_timesteps,1)
        scale: TensorShape(batch,1,1)
        '''
        if self.stat_variables:
            static_vars_list, decoder_vars_list, mask, scale = inputs
        else:
            decoder_vars_list, mask, scale = inputs
            
        # scale
        scale = scale[:,-1:,:]
        s_dim = scale.shape.as_list()[-1] 
        
        if s_dim == 2:
            scaler = 'standard_scaler'
            scale_mean = scale[:,:,0:1]
            scale_std = scale[:,:,1:2]
        else:
            scaler = 'mean_scaler'
            scale_mean = scale[:,:,0:1]
            scale_std = scale[:,:,0:1]
        
        # static var selection 
        if self.stat_variables:
            static_vec, static_weights = self.static_input_layer(static_vars_list, training=training)
            context = self.static_context_layer(static_vec, training=training)
            enrichment_vec = self.enrich_vector_layer(static_vec, training=training)
            init_h, init_c = self.init_states_layer(static_vec)            
            # dec input prep
            target_d, decoder_weights = self.decoder_input_layer([decoder_vars_list, context], training=training)

        else:
            static_weights = None
            target_d, decoder_weights = self.decoder_input_layer(decoder_vars_list, training=training)
            init_h, init_c = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32), tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            
        # lstm init states
        init_states = [init_h, init_c]
        
        # recurrent_layer
        temporal_features = self.recurrent_layer([target_d, init_states], training=training)
        
        # static feature enrichment
        if self.stat_variables:
            enriched_features = self.static_enrich_layer([temporal_features, enrichment_vec], training=training)
        else:
            enriched_features = self.static_enrich_layer(temporal_features, training=training)
        
        # masks
        padding_mask = create_padding_mask(mask[:,:self.seq_len,0])
        look_ahead_mask = create_look_ahead_mask(self.seq_len)
        
        # attention stack
        dec_output, _ = self.decoder(enriched_features, look_ahead_mask, padding_mask, training=training)
        
        # final_gating_layer
        dec_output = self.pff_layer([dec_output, temporal_features], training=training)
        
        dec_output = dec_output[:,-self.f_len:,:]
        decoder_weights = decoder_weights[:, -self.f_len:, :]
        
        if self.loss_fn == 'Point':
            out = self.final_layer(dec_output, training=training)
        elif self.loss_fn == 'Poisson':
            if scaler == 'mean_scaler':
                mean = self.final_layer(dec_output, training=training)*scale
            elif scaler == 'standard_scaler':
                mean = self.final_layer(dec_output, training=training)*scale_std + scale_mean
            parameters = mean
            out = poisson_sample(mu=mean)
        elif self.loss_fn == 'Quantile':
            if self.num_quantiles > 1:
                out = tf.math.cumsum(tf.concat([self.proj_intrcpt(dec_output, training=training), self.proj_incrmnt(dec_output, training=training)], axis=-1), axis=-1)
            else:
                out = self.proj_intrcpt(dec_output, training=training)
        elif self.loss_fn == 'Normal':
            if scaler == 'mean_scaler':
                mean = self.m_layer(dec_output, training=training)*scale       # (batch, f_len, 1)
                stddev = self.s_layer(dec_output, training=training)*scale      # (batch, f_len, 1)
            elif scaler == 'standard_scaler':
                mean = self.m_layer(dec_output, training=training)*scale_std + scale_mean       # (batch, f_len, 1)
                stddev = self.s_layer(dec_output, training=training)*scale_std     # (batch, f_len, 1)     
            parameters = tf.concat([mean, stddev], axis=-1)
            out = normal_sample(mean, stddev)
        elif self.loss_fn == 'Negbin':
            if scaler == 'mean_scaler':
                mean = self.m_layer(dec_output, training=training)*scale       # (batch, f_len, 1)
                alpha = self.a_layer(dec_output, training=training)*tf.sqrt(scale)      # (batch, f_len, 1)
            elif scaler == 'standard_scaler':
                mean = self.m_layer(dec_output, training=training)*scale_std + scale_mean       # (batch, f_len, 1)
                alpha = self.a_layer(dec_output, training=training)*tf.sqrt(scale_std)      # (batch, f_len, 1)
            parameters = tf.concat([mean, alpha], axis=-1)
            out = negbin_sample(mean, alpha)
        else:
            raise ValueError('Invalid Loss Function.')
        
        #print("decoder op shape: ", dec_output.shape)
        #print("decoder wts shape: ", decoder_weights.shape)
        if self.loss_fn == 'Point' or self.loss_fn == 'Quantile':
            return out, scale, static_weights, decoder_weights
        else:
            return out, parameters, static_weights, decoder_weights


# VarTransformer Wrapper

class SageTransformer_Model(tf.keras.Model):
    def __init__(self,
                 col_index_dict,
                 vocab_dict,
                 num_layers,
                 num_heads,
                 kernel_sizes,
                 d_model,
                 forecast_horizon,
                 max_inp_len,
                 loss_type,
                 num_quantiles,
                 dropout_rate=0.1):

        super(SageTransformer_Model, self).__init__()
        
        self.hist_len = int(max_inp_len-1)
        self.f_len = int(forecast_horizon)
        self.loss_type = loss_type
        self.col_index_dict = col_index_dict
        self.vocab_dict = vocab_dict
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.max_inp_len = max_inp_len
        self.dropout_rate = dropout_rate
        
        if (len(self.col_index_dict.get('static_num_indices')[0])==0) and (len(self.col_index_dict.get('static_cat_indices')[0])==0):
            self.static_variables = False
        else:
            self.static_variables = True
        
        self.model = SageTransformer(static_variables = self.static_variables,
                                    num_layers = num_layers,
                                    d_model = d_model,
                                    num_heads = num_heads, 
                                    kernel_sizes = kernel_sizes,
                                    dff = d_model,
                                    hist_len = self.hist_len, 
                                    f_len = self.f_len,
                                    loss_fn = self.loss_type,
                                    num_quantiles = num_quantiles,
                                    rate = dropout_rate)
        
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
                                                                            output_dim = d_model,
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
                                                                                      output_dim = d_model,
                                                                                      name = "embedding_layer_{}".format(colname))
        
        # columns names & indices
        self.target_col_name, self.target_index = self.col_index_dict.get('target_index')
        self.stat_num_col_names, self.stat_num_indices = self.col_index_dict.get('static_num_indices')
        self.stat_cat_col_names, self.stat_cat_indices = self.col_index_dict.get('static_cat_indices')
        self.known_num_col_names, self.known_num_indices = self.col_index_dict.get('temporal_known_num_indices')
        self.unknown_num_col_names, self.unknown_num_indices = self.col_index_dict.get('temporal_unknown_num_indices')
        self.known_cat_col_names, self.known_cat_indices = self.col_index_dict.get('temporal_known_cat_indices')
        self.unknown_cat_col_names, self.unknown_cat_indices = self.col_index_dict.get('temporal_unknown_cat_indices')
        
        # Create Numerical Embedding (Linear Transform) Layers
        
        self.target_linear_transform_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=d_model, use_bias=False))
        #self.hist_encode_layer = tf.keras.layers.LSTM(units=d_model, return_state=False, return_sequences=True) 
      
        if len(self.stat_num_col_names)>0:
            self.stat_linear_transform_layers = {}
            for colname in self.stat_num_col_names:
                self.stat_linear_transform_layers[colname] = tf.keras.layers.Dense(units=d_model, use_bias=False)
        
        if len(self.known_num_col_names)>=0:
            self.known_linear_transform_layers = {}
            for colname in self.known_num_col_names + ['rel_age']:
                self.known_linear_transform_layers[colname] = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=d_model, use_bias=False))
                
    def call(self, inputs, training):
        
        # total_dim
        t_dim = inputs.shape.as_list()[-1] - 3 # reduce 2 dims for mask,rel_age,scale
        dim_counter = 0
        
        # target
        target = tf.strings.to_number(inputs[:,:,self.target_index:self.target_index+1], out_type=tf.dtypes.float32)
        dim_counter += 1
        
        # transform to model dim
        target = self.target_linear_transform_layer(target)
        
        # ordered col names list
        stat_cols_ordered_list = []
        future_cols_ordered_list = []
        
        # stat, encoder, decoder tensor lists
        static_vars_list = []
        decoder_vars_list = [target[:,:-1,:]] # left-shifted actuals
        future_cols_ordered_list = future_cols_ordered_list + [self.target_col_name]
        
        # recurrent state encoding w/ causal conv encoder
        #recurrent_state = self.hist_encode_layer(target[:,:-1,:]) # (batch, seq_len-1, d_model)
        #decoder_vars_list.append(recurrent_state) # encoded state till current timestep
        #future_cols_ordered_list = future_cols_ordered_list + ['{}_past_encoding'.format(self.target_col_name)]
        
        # static numeric
        if len(self.stat_num_indices)>0:
            stat_cols_ordered_list = stat_cols_ordered_list + self.stat_num_col_names
            for col, i in zip(self.stat_num_col_names, self.stat_num_indices):
                stat_var = tf.strings.to_number(inputs[:,-1,i:i+1], out_type=tf.dtypes.float32)
                stat_var = self.stat_linear_transform_layers[col](stat_var)
                # append
                static_vars_list.append(stat_var)
                dim_counter += 1
        
        # static embeddings
        if len(self.stat_cat_indices)>0:
            stat_cols_ordered_list = stat_cols_ordered_list + self.stat_cat_col_names
            for col, i in zip(self.stat_cat_col_names, self.stat_cat_indices):
                stat_var = inputs[:,:,i]
                stat_var_id = self.stat_lookup_tables.get(col).lookup(stat_var)
                stat_var_embeddings = self.stat_embed_layers.get(col)(stat_var_id)
                # append
                static_vars_list.append(stat_var_embeddings[:,-1,:])
                dim_counter += 1
                
        # known numeric 
        if len(self.known_num_indices)>0:
            future_cols_ordered_list = future_cols_ordered_list + self.known_num_col_names
            for col, i in zip(self.known_num_col_names, self.known_num_indices):
                num_vars = tf.strings.to_number(inputs[:,1:,i:i+1], out_type=tf.dtypes.float32)
                num_vars = self.known_linear_transform_layers[col](num_vars)
                # append
                decoder_vars_list.append(num_vars)
                dim_counter += 1
                    
        # known embeddings
        if len(self.known_cat_indices)>0:
            future_cols_ordered_list = future_cols_ordered_list + self.known_cat_col_names
            for col, i in zip(self.known_cat_col_names, self.known_cat_indices):
                cat_var = inputs[:,1:,i]
                cat_var_id = self.temporal_known_lookup_tables.get(col).lookup(cat_var)
                cat_var_embeddings = self.temporal_known_embed_layers.get(col)(cat_var_id)
                # append
                decoder_vars_list.append(cat_var_embeddings)
                dim_counter += 1
                
        # unsued cols for this model
        if len(self.unknown_num_indices)>0:
            for col, i in zip(self.unknown_num_col_names, self.unknown_num_indices):
                dim_counter += 1
        
        if len(self.unknown_cat_indices)>0:
            for col, i in zip(self.unknown_cat_col_names, self.unknown_cat_indices):
                dim_counter += 1
                
        # remaining_dim
        r_dim = t_dim - dim_counter

        # default scale
        scale = tf.strings.to_number(inputs[:,:-1,-2:-1], out_type=tf.dtypes.float32)
        
        if r_dim == 2: # standard scaling used (mean,std)
            #print(" standard r_dim")
            # rel_age
            rel_age = tf.strings.to_number(inputs[:,:-1,-4:-3], out_type=tf.dtypes.float32)
            rel_age_dec = self.known_linear_transform_layers['rel_age'](rel_age)
            # append
            decoder_vars_list.append(rel_age_dec)
            # scale
            scale = tf.strings.to_number(inputs[:,:-1,-3:-1], out_type=tf.dtypes.float32)
            #scale_log = scale[:,:,0:1] #tf.math.log(tf.math.sqrt(scale[:,:,0:1]))
            #scale_log = self.scale_linear_transform_layer(scale_log[:,-1,:])
            # append
            #static_vars_list.append(scale_log)
        elif r_dim == 1:
            #print(" mean r_dim")
             # rel_age
            rel_age = tf.strings.to_number(inputs[:,:-1,-3:-2], out_type=tf.dtypes.float32)
            rel_age_dec = self.known_linear_transform_layers['rel_age'](rel_age)
            # append
            decoder_vars_list.append(rel_age_dec)
            # scale
            scale = tf.strings.to_number(inputs[:,:-1,-2:-1], out_type=tf.dtypes.float32)
            #scale_log = tf.math.log(tf.math.sqrt(scale))
            #scale_log = self.scale_linear_transform_layer(scale_log[:,-1,:])
            # append
            #static_vars_list.append(scale_log)
            
        # Append additional columns
        #stat_cols_ordered_list = stat_cols_ordered_list + ['scale']
        future_cols_ordered_list = future_cols_ordered_list + ['rel_age']
        
        # mask
        mask = tf.strings.to_number(inputs[:,:-1,-1:], out_type=tf.dtypes.float32)
        
        
        # model process
        if self.static_variables:
            o, s, s_wts, f_wts = self.model([static_vars_list, decoder_vars_list, mask, scale], training=training)
        else:
            o, s, s_wts, f_wts = self.model([decoder_vars_list, mask, scale], training=training)

        # Retain period-wise importance
        bs = tf.shape(f_wts)[0]
        f_wts = tf.reshape(f_wts, [bs*(self.f_len), -1])
              
        return o, s, ([stat_cols_ordered_list,future_cols_ordered_list], [s_wts, f_wts])
    
    
def SageTransformer_Train(model,
                      target_index,
                      train_dataset, 
                      test_dataset, 
                      loss_type,
                      loss_function, 
                      metric, 
                      learning_rate,
                      max_epochs, 
                      min_epochs,
                      prefill_buffers,
                      num_train_samples,
                      num_test_samples,
                      train_batch_size,
                      test_batch_size,
                      train_steps_per_epoch,
                      test_steps_per_epoch,
                      patience,
                      weighted_training,
                      model_prefix,
                      logdir,
                      opt=None,
                      clipnorm=None,
                      min_delta=0.0001,
                      shuffle=True):
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
            o, s, f = model(x_train, training=training)
            out_len = tf.shape(s)[1]
            s_dim = tf.shape(scale)[-1]
            if loss_type in ['Normal','Poisson','Negbin']:
                if s_dim == 1:
                    if weighted_training:
                        loss = loss_function(y_train*scale[:,-out_len:,:], [s, wts])
                    else:
                        loss = loss_function(y_train*scale[:,-out_len:,:], s)
                else:
                    s_mean = scale[:,-out_len:,0:1]
                    s_std = scale[:,-out_len:,1:2]
                    if weighted_training:
                        loss = loss_function(y_train*s_std + s_mean, [s, wts])
                    else:
                        loss = loss_function(y_train*s_std + s_mean, s)
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
        out_len = tf.shape(y_test)[1]
        s_dim = tf.shape(scale)[-1]
        hist_len = tf.shape(x_test)[1] - out_len
        
        if s_dim == 1:
            scale_mean = scale[:,-1:,-1]
        else:
            scale_mean = scale[:,-1:,0]
            scale_std = scale[:,-1:,1]
            
        # recursive predict
        output = []
        
        for i in range(out_len):
            o, s, f = model(x_test, training=training)
            
            # update target
            if loss_type in ['Normal','Poisson','Negbin']:
                s = s.numpy()
                output.append(s[:,i:i+1,:])
                infer_arr = x_test.numpy()
                if s_dim == 1:
                    infer_arr[:,hist_len:hist_len+i+1,target_index] = o[:,0:i+1,0]/scale_mean
                else:
                    infer_arr[:,hist_len:hist_len+i+1,target_index] = np.nan_to_num((o[:,0:i+1,0] - scale_mean)/scale_std)
            
            elif loss_type in ['Point','Quantile']:
                o = o.numpy()
                output.append(o[:,i:i+1,:])
                infer_arr = x_test.numpy()
                infer_arr[:,hist_len:hist_len+i+1,target_index] = o[:,0:i+1,0] # append q=0.5 value assuming it's the first in sequence

            # feedback updated hist + fh tensor
            x_test = tf.convert_to_tensor(np.char.decode(x_test.astype(np.bytes_), 'UTF-8'), dtype=tf.string) 
        
        o = tf.convert_to_tensor(np.concatenate(output, axis=1))
        
        if loss_type in ['Normal','Poisson','Negbin']:
            if s_dim == 1:
                if weighted_training:
                    loss = loss_function(y_test*scale[:,-out_len:,:], [o, wts])
                else:
                    loss = loss_function(y_test*scale[:,-out_len:,:], o)
            else:
                s_mean = scale[:,-out_len:,0:1]
                s_std = scale[:,-out_len:,1:2]
                if weighted_training:
                    loss = loss_function(y_test*s_std + s_mean, [o, wts])
                else:
                    loss = loss_function(y_test*s_std + s_mean, o)
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

    model_tracker_file.write('Feature Weighted ConvTransformer Training started with following Model Parameters ... \n')
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
    
    ######################################################### train loop -- pre-filled tensor buffers
    
    def shuffle_arrays(arrays, set_seed=-1):
      """Shuffles arrays in-place, in the same order, along axis=0
      arrays : List of NumPy arrays.
      set_seed : Seed value if int >= 0, else seed is random.
      """
      assert all(len(arr) == len(arrays[0]) for arr in arrays)
      seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed
      for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)
    
    if prefill_buffers:
      print("prefetching training samples ... ")
      # get batch size
      batch_size = 0
      for x,y,s,w in train_dataset.take(1):
        batch_size = int(x.shape[0])

      x_train = []
      y_train = []
      train_scale = []
      train_wts = []
      #num_train_batches = 0
      for step, (x_batch, y_batch, scale, wts) in enumerate(train_dataset):
        x_train.append(x_batch)
        y_train.append(y_batch)
        train_scale.append(scale)
        train_wts.append(wts)
        #num_train_batches = num_train_batches + m.floor(batch_size/train_batch_size)
        if (step+1)*batch_size >= num_train_samples:
          break
       
      # concat
      x_train = tf.concat(x_train, axis=0)
      y_train = tf.concat(y_train, axis=0)
      train_scale = tf.concat(train_scale, axis=0)
      train_wts = tf.concat(train_wts, axis=0)
      print("Training Samples Gathered: ", x_train.shape[0])
      
      print("prefetching test samples ... ")
      x_test = []
      y_test = []
      test_scale = []
      test_wts = []
      #num_test_batches = 0
      for step, (x_batch, y_batch, scale, wts) in enumerate(test_dataset):
        x_test.append(x_batch)
        y_test.append(y_batch)
        test_scale.append(scale)
        test_wts.append(wts)
        #num_test_batches = num_test_batches + m.floor(batch_size/train_batch_size)
        if (step+1)*batch_size >= num_test_samples:
          break

      # concat
      x_test = tf.concat(x_test, axis=0)
      y_test = tf.concat(y_test, axis=0)
      test_scale = tf.concat(test_scale, axis=0)
      test_wts = tf.concat(test_wts, axis=0)
      print("Test Samples Gathered: ", x_test.shape[0])
        
      num_train_batches = int(x_train.shape[0]//train_batch_size)
      num_test_batches = int(x_test.shape[0]//test_batch_size)

      for epoch in range(max_epochs):
          print("Epoch {}/{}". format(epoch, max_epochs))
          # shuffle Training data only,if shuffle=True
          if shuffle:
            #shuffle_arrays([x_train, y_train, train_scale, train_wts])
            indices = tf.range(start=0, limit=tf.shape(x_train)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            x_train = tf.gather(x_train, shuffled_indices)
            y_train = tf.gather(y_train, shuffled_indices)
            train_scale = tf.gather(train_scale, shuffled_indices)
            train_wts = tf.gather(train_wts, shuffled_indices)

          for i in range(num_train_batches):
            x_batch = x_train[i*train_batch_size:(i+1)*train_batch_size]
            y_batch = y_train[i*train_batch_size:(i+1)*train_batch_size]
            scale = train_scale[i*train_batch_size:(i+1)*train_batch_size]
            wts = train_wts[i*train_batch_size:(i+1)*train_batch_size]
            train_loss, train_out = trainstep(model, optimizer, x_batch, y_batch, scale, wts, training=True)
            out_len = tf.shape(train_out)[1]
            train_loss_avg.update_state(train_loss)
            if loss_type in ['Normal','Poisson','Negbin']:
              train_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], train_out)
            elif loss_type in ['Point','Quantile']:
              train_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], train_out*scale[:,-out_len:,:])
            with train_summary_writer.as_default():
              tf.summary.scalar('loss', train_loss_avg.result(), step=(i+1)*(epoch+1))
              tf.summary.scalar('accuracy', train_metric.result(), step=(i+1)*(epoch+1))

          for i in range(num_test_batches):
            x_batch = x_test[i*train_batch_size:(i+1)*train_batch_size]
            y_batch = y_test[i*train_batch_size:(i+1)*train_batch_size]
            scale = test_scale[i*train_batch_size:(i+1)*train_batch_size]
            wts = test_wts[i*train_batch_size:(i+1)*train_batch_size]
            test_loss, test_out = trainstep(model, optimizer, x_batch, y_batch, scale, wts, training=False)
            out_len = tf.shape(test_out)[1]
            test_loss_avg.update_state(test_loss)
            if loss_type in ['Normal','Poisson','Negbin']:
              test_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], test_out)
            elif loss_type in ['Point','Quantile']:
              test_metric.update_state(y_batch[:,-out_len:,:]*scale[:,-out_len:,:], test_out*scale[:,-out_len:,:])
            with test_summary_writer.as_default():
              tf.summary.scalar('loss', test_loss_avg.result(), step=(i+1)*(epoch+1))
              tf.summary.scalar('accuracy', test_metric.result(), step=(i+1)*(epoch+1))

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

          prev_min_loss = np.min(test_loss_results[:-1])
          current_min_loss = np.min(test_loss_results)
          delta = current_min_loss - prev_min_loss

          print("Improvement delta (min_delta {}):  {}".format(min_delta, delta))
          # track & save best model
          if test_loss_results[epoch]==np.min(test_loss_results) and (delta > min_delta):
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
    
    else:
      # Use random, dynamic samples from generator 
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
              if step > test_steps_per_epoch:
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
          
          prev_min_loss = np.min(test_loss_results[:-1])
          current_min_loss = np.min(test_loss_results)
          delta = current_min_loss - prev_min_loss
          
          print("Improvement delta (min_delta {}):  {}".format(min_delta, delta))
          # track & save best model
          if (test_loss_results[epoch]==np.min(test_loss_results)) and (delta > min_delta):
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
        
 
        
def SageTransformer_InferRecursive(model, inputs, loss_type, hist_len, f_len, target_index, num_quantiles):
    infer_tensor, scale, id_arr, date_arr = inputs
    s_dim = scale.shape[-1]
    
    if s_dim == 1:
        scale = scale[:,-1:,-1]
    else:
        scale_mean = scale[:,-1:,0]
        scale_std = scale[:,-1:,1]
        
    window_len = hist_len + f_len
    output = []
    stat_wts_df = None 
    decoder_wts_df = None
        
    for i in range(f_len):
        out, dist, feature_wts = model(infer_tensor, training=False)
            
        # update target
        if loss_type in ['Normal','Poisson','Negbin']:
            dist = dist.numpy()
            output.append(dist[:,i:i+1,:])
            infer_arr = infer_tensor.numpy()
            if s_dim == 1:
                infer_arr[:,hist_len:hist_len+i+1,target_index] = out[:,0:i+1,0]/scale
            else:
                infer_arr[:,hist_len:hist_len+i+1,target_index] = np.nan_to_num((out[:,0:i+1,0] - scale_mean)/scale_std)
        elif loss_type in ['Point','Quantile']:
            out = out.numpy()
            output.append(out[:,i:i+1,:])
            infer_arr = infer_tensor.numpy()
            infer_arr[:,hist_len+i:hist_len+i+1,target_index] = out[:,i:i+1,0] # feedback q=0.5 percentile
            
        # feedback updated hist + fh tensor
        infer_tensor = tf.convert_to_tensor(np.char.decode(infer_arr.astype(np.bytes_), 'UTF-8'), dtype=tf.string)
            
        if i == (f_len - 1):
            column_names_list, wts_list = feature_wts
            stat_columns, decoder_columns = column_names_list
            stat_columns_string = []
            decoder_columns_string = []
            for col in stat_columns:
                stat_columns_string.append(col.numpy().decode("utf-8")) 
            for col in decoder_columns:
                decoder_columns_string.append(col.numpy().decode("utf-8"))
                
            #print(" decoder_columns_string: ",  decoder_columns_string)
            stat_wts, decoder_wts = wts_list
            # Average feature weights across time dim
            decoder_wts = decoder_wts.numpy()
            # convert wts to df    
            decoder_wts_df = pd.DataFrame(decoder_wts, columns=decoder_columns_string)    
            if stat_wts is not None:
                stat_wts = stat_wts.numpy()
                stat_wts_df = pd.DataFrame(stat_wts, columns=stat_columns_string)   
            
    output_arr = np.concatenate(output, axis=1) 
    
    # rescale if necessary
    if loss_type in ['Normal','Poisson','Negbin']:
        output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1),output_arr), axis=1))
        output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value':'forecast'})
        output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
    elif loss_type in ['Point']:
        # 
        if s_dim == 1:
            output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1),output_arr*scale.reshape(-1,1)), axis=1))
        else:
            output_arr = output_arr*scale_std.reshape(-1,1) + scale_mean.reshape(-1,1)
            output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1), output_arr), axis=1))
        output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value':'forecast'})
        output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
            
    elif loss_type in ['Quantile']:
        df_list = []
        for i in range(num_quantiles):
            if s_dim == 1:
                output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1),output_arr[:,:,i]*scale.reshape(-1,1)), axis=1))
                output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value': f"forecast_{i}"})
                output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
                df_list.append(output_df)
            else:
                output = output_arr[:,:,i]* scale_std.reshape(-1, 1) + scale_mean.reshape(-1, 1)
                output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1, 1), output), axis=1))
                output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0: 'id', 'value': f"forecast_{i}"})
                output_df = output_df.rename_axis('index').sort_values(by=['id', 'index']).reset_index(drop=True)
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
        
    # weights df merge with id
    stat_wts_df = pd.concat([pd.DataFrame(id_arr.reshape(-1,1)), stat_wts_df], axis=1)
    
    decoder_wts_df = pd.concat([date_df, decoder_wts_df], axis=1)
    fid_df = pd.DataFrame(np.repeat(id_arr.reshape(-1,1), f_len, axis=0))
    fid_df.columns = ['id']
    decoder_wts_df = pd.concat([fid_df, decoder_wts_df], axis=1) 
    
    print(stat_wts_df.shape, decoder_wts_df.shape)
        
    return forecast_df, stat_wts_df, decoder_wts_df


class SageModel:
    def __init__(self, 
                 col_index_dict,
                 vocab_dict,
                 num_layers,
                 num_heads,
                 kernel_sizes,
                 d_model,
                 forecast_horizon,
                 max_inp_len,
                 loss_type,
                 num_quantiles,
                 dropout_rate=0.1):
        
        self.col_index_dict = col_index_dict
        self.vocab_dict = vocab_dict
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kernel_sizes = kernel_sizes
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.max_inp_len = max_inp_len
        self.loss_type = loss_type
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        self.target_col_name, self.target_index = self.col_index_dict.get('target_index')
    
    def build(self):
        tf.keras.backend.clear_session()
        self.model = SageTransformer_Model(self.col_index_dict,
                                  self.vocab_dict,
                                  self.num_layers,
                                  self.num_heads,
                                  self.kernel_sizes,
                                  self.d_model,
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
              prefill_buffers,
              num_train_samples,
              num_test_samples,
              train_batch_size,
              test_batch_size,
              train_steps_per_epoch,
              test_steps_per_epoch,
              patience,
              weighted_training,
              model_prefix,
              logdir,
              load_model=None,
              opt=None,
              clipnorm=None,
              min_delta=0.0001,
              shuffle=True):
        
        if load_model is None:
            # Initialize Weights
            for x,y,s,w in train_dataset.take(1):
                self.model(x, training=False)
        else:
            # Initialize Weights
            for x,y,s,w in train_dataset.take(1):
                self.model(x, training=False)
            saved_model = tf.keras.models.load_model(load_model)
            self.model.set_weights(saved_model.get_weights())
            del saved_model
            gc.collect()
            print("Saved model: {} loaded. Continuing training ...".format(load_model))
        
        best_model = SageTransformer_Train(self.model,
                                          self.target_index,    
                                          train_dataset, 
                                          test_dataset, 
                                          self.loss_type,
                                          loss_function, 
                                          metric, 
                                          learning_rate,
                                          max_epochs, 
                                          min_epochs,
                                          prefill_buffers,
                                          num_train_samples,
                                          num_test_samples,
                                          train_batch_size,
                                          test_batch_size,
                                          train_steps_per_epoch,
                                          test_steps_per_epoch,
                                          patience,
                                          weighted_training,
                                          model_prefix,
                                          logdir,
                                          opt,
                                          clipnorm,
                                          min_delta,
                                          shuffle)
        return best_model
    
    def load(self, model_path):
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model(model_path)
        
    def infer(self, inputs):
        forecast, stat_wts_df, decoder_wts_df = SageTransformer_InferRecursive(self.model, inputs, self.loss_type, self.max_inp_len, self.forecast_horizon, self.target_index, self.num_quantiles)
        
        return forecast, [stat_wts_df, decoder_wts_df]
    
    

