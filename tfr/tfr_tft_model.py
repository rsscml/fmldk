#!/usr/bin/env python
# coding: utf-8

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


# In[3]:


# Self-Attention
def ScaledDotProductAttention(q, k, v, causal_mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # add the mask to the scaled tensor.
    if causal_mask is not None:
        scaled_attention_logits += (causal_mask * -1e9)
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

def create_padding_mask(seq):
    seq = tf.cast(tf.math.less(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, :]  # (batch_size, 1, seq_len)
  
def causal_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class TFTMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_head, d_model, dropout_rate):
        super(TFTMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout_rate = dropout_rate

        assert d_model % self.n_head == 0

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # Use same value layer to facilitate interpretation
        vs_layer = tf.keras.layers.Dense(units=d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(tf.keras.layers.Dense(units=d_k, use_bias=False))
            self.ks_layers.append(tf.keras.layers.Dense(units=d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.w_o = tf.keras.layers.Dense(units=d_model, use_bias=False)

    def call(self, q, k, v, causal_mask, training):
        """
        q: Query tensor of shape=(?, T, d_model)
        k: Key of shape=(?, T, d_model)
        v: Values of shape=(?, T, d_model)
        mask: Masking if required with shape=(?, T, T)
        Returns:
        Tuple of (layer outputs, attention weights)
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = ScaledDotProductAttention(qs, ks, vs, causal_mask)
            head_dropout = tf.keras.layers.Dropout(self.dropout_rate)(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = tf.stack(heads, axis=0) if n_head > 1 else heads[0]
        attn = tf.stack(attns, axis=0)

        outputs = tf.math.reduce_mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = tf.keras.layers.Dropout(self.dropout_rate)(outputs)

        return outputs, attn

# GRN
class tft_linear_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, activation=None, use_time_distributed=False, use_bias=True):
        super(tft_linear_layer, self).__init__()
        self.linear = tf.keras.layers.Dense(units=hidden_layer_size, activation=activation, use_bias=use_bias)
        if use_time_distributed:
            self.linear = tf.keras.layers.TimeDistributed(self.linear)

    def call(self, inputs, training):
        return self.linear(inputs, training=training)


class tft_apply_mlp(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, output_activation=None, hidden_activation='tanh', use_time_distributed=False):
        super(tft_apply_mlp, self).__init__()
        if use_time_distributed:
            self.hidden_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation=hidden_activation))
            self.out_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size, activation=output_activation))
        else:
            self.hidden_layer = tf.keras.layers.Dense(hidden_size, activation=hidden_activation)
            self.out_layer = tf.keras.layers.Dense(output_size, activation=output_activation)

    def call(self,inputs, training):
        x = self.hidden_layer(inputs, training=training)
        return self.out_layer(x, training=training)


class tft_apply_gating_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate=None, use_time_distributed=True, activation=None):
        super(tft_apply_gating_layer, self).__init__()
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


class tft_add_and_norm_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(tft_add_and_norm_layer, self).__init__()
        self.Add = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training):
        tmp = self.Add(inputs)
        tmp = self.LayerNorm(tmp)
        return tmp


class tft_grn_layer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_layer_size,
                 output_size=None,
                 dropout_rate=None,
                 use_time_distributed=True,
                 additional_context=False,
                 return_gate=False):
        super(tft_grn_layer, self).__init__()

        self.hidden_layer_size = hidden_layer_size

        if output_size is None:
            self.output_size = hidden_layer_size
        else:
            self.output_size = output_size

        self.linear = tf.keras.layers.Dense(self.output_size)
        if use_time_distributed:
                self.linear = tf.keras.layers.TimeDistributed(self.linear)

        self.additional_context = additional_context
        self.return_gate = return_gate

        self.hidden_1 = tft_linear_layer(hidden_layer_size=hidden_layer_size,
                                         activation=None,
                                         use_time_distributed=use_time_distributed)

        if additional_context:
            self.context_layer = tft_linear_layer(hidden_layer_size=hidden_layer_size,
                                                  activation=None,
                                                  use_time_distributed=use_time_distributed,
                                                  use_bias=False)

        self.hidden_2 = tft_linear_layer(hidden_layer_size=hidden_layer_size,
                                         activation=None,
                                         use_time_distributed=use_time_distributed)

        self.gate = tft_apply_gating_layer(hidden_layer_size=self.output_size,
                                           dropout_rate=dropout_rate,
                                           use_time_distributed=use_time_distributed,
                                           activation=None)

        self.add_norm = tft_add_and_norm_layer()

    def call(self, inputs, training):

        if self.additional_context:
            x, c = inputs
            skip = self.linear(x, training=training)
            hidden = self.hidden_1(x, training=training)
            hidden = hidden + self.context_layer(c, training=training)
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
            return self.add_norm([skip, gating_layer]), gate
        else:
            return self.add_norm([skip, gating_layer])


# variable selection networks
class variable_selection_static(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate):
        super(variable_selection_static, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.num_vars = len(input_shape)
        self.grn_flat = tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                   output_size=self.num_vars,
                                   dropout_rate=self.dropout_rate,
                                   use_time_distributed=False,
                                   additional_context=None,
                                   return_gate=False)
        self.grn_var = [tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                    output_size=None,
                                    dropout_rate=self.dropout_rate,
                                    use_time_distributed=False,
                                    additional_context=None,
                                    return_gate=False) for _ in range(self.num_vars)]

    def call(self, inputs, training):

        # inputs: list of static vars [[Batch, var_dim], ...]

        flatten = tf.concat(inputs, axis=1) #[batch,sum_of_var_dims]
        mlp_outputs = self.grn_flat(flatten, training=training)
        
        static_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)  #[batch, num_vars]
        weights = tf.expand_dims(static_weights, axis=-1)                    #[batch, num_vars, 1]

        trans_emb_list = []
        for i in range(self.num_vars):
            e = self.grn_var[i](inputs[i], training=training)
            trans_emb_list.append(e)

        trans_embedding = tf.stack(trans_emb_list, axis=1) #[batch, num_vars, hidden_layer_size]
        
        combined = tf.keras.layers.Multiply()([weights, trans_embedding])
        static_vec = tf.math.reduce_sum(combined, axis=1) #[batch, hidden_layer_size]
        
        return static_vec, static_weights


class static_contexts(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate):
        super(static_contexts, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.stat_vec_layer = tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                            output_size=None,
                                            dropout_rate=self.dropout_rate,
                                            use_time_distributed=False,
                                            additional_context=None,
                                            return_gate=False)

        self.enrich_vec_layer = tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                            output_size=None,
                                            dropout_rate=self.dropout_rate,
                                            use_time_distributed=False,
                                            additional_context=None,
                                            return_gate=False)

        self.h_vec_layer = tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                         output_size=None,
                                         dropout_rate=self.dropout_rate,
                                         use_time_distributed=False,
                                         additional_context=None,
                                         return_gate=False)

        self.c_vec_layer = tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                         output_size=None,
                                         dropout_rate=self.dropout_rate,
                                         use_time_distributed=False,
                                         additional_context=None,
                                         return_gate=False)

    def call(self, inputs, training):
        # inputs: static_vec
        static_vec = self.stat_vec_layer(inputs, training=training)

        enrich_vec = self.enrich_vec_layer(inputs, training=training)

        h_vec = self.h_vec_layer(inputs, training=training)

        c_vec = self.c_vec_layer(inputs, training=training)

        return static_vec, enrich_vec, h_vec, c_vec


class variable_selection_temporal(tf.keras.layers.Layer):
    """
    Takes inputs as a list of list of embedded/linear transformed tensors of shape [batch,time_steps,emb_dim] & context vec
    """
    def __init__(self, hidden_layer_size, static_context, dropout_rate):
        super(variable_selection_temporal, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.context = static_context
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if self.context:
            self.num_vars = len(input_shape[0])
        else:
            self.num_vars = len(input_shape)

        self.grn_flat = tft_grn_layer(hidden_layer_size = self.hidden_layer_size,
                                      output_size = self.num_vars,
                                      dropout_rate=self.dropout_rate,
                                      use_time_distributed=True,
                                      additional_context=self.context,
                                      return_gate=False)

        self.grn_var = [tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                      output_size=None,
                                      dropout_rate=self.dropout_rate,
                                      use_time_distributed=False,
                                      additional_context=None,
                                      return_gate=False) for _ in range(self.num_vars)]

    def call(self, x, training):
        # x: [[[batch,timesteps,var_dim], ...], context_vector]

        if self.context:
            inputs, context = x
            num_vars = len(inputs)
            context = tf.expand_dims(context, axis=1)     #[batch, 1, context_dim]
            flatten = tf.concat(inputs, axis=-1)          #[batch, time_steps, sum_of_var_dims]
            mlp_outputs = self.grn_flat([flatten, context], training=training)
        else:
            inputs = x
            num_vars = len(inputs)
            flatten = tf.concat(inputs, axis=-1)          #[batch, time_steps, sum_of_var_dims]
            mlp_outputs = self.grn_flat(flatten, training=training)
        
        dynamic_weights = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('softmax'))(mlp_outputs) #[batch, T, num_vars]
        #weights = tf.expand_dims(weights, axis=2)       #[batch, T, 1, num_vars]
        weights = tf.expand_dims(dynamic_weights, axis=-1)       #[batch, T, num_vars, 1]

        trans_emb_list = []
        for i in range(self.num_vars):
            e = self.grn_var[i](inputs[i], training=training)
            trans_emb_list.append(e)

        #trans_embedding = tf.stack(trans_emb_list, axis=-1)  #[batch, T, hidden_layer_size, num_vars]
        trans_embedding = tf.stack(trans_emb_list, axis=2) #[batch,T,num_vars,hidden_layer_size]
        
        combined = tf.keras.layers.Multiply()([weights, trans_embedding])
        #lstm_input = tf.reduce_sum(combined, axis=-1)
        lstm_input = tf.reduce_sum(combined, axis=2) #[batch,T,hidden_layers_size]
        
        return lstm_input, dynamic_weights


# LSTM Layer

class lstm_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, rnn_layers, dropout_rate):
        super(lstm_layer, self).__init__()

        self.rnn_layers = rnn_layers
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        self.tft_encoder = tf.keras.layers.LSTM(units=hidden_layer_size, return_state=True, return_sequences=True)
        self.tft_decoder = tf.keras.layers.LSTM(units=hidden_layer_size, return_state=False, return_sequences=True)

        self.gate = tft_apply_gating_layer(hidden_layer_size=self.hidden_layer_size,
                                           dropout_rate=self.dropout_rate,
                                           use_time_distributed=True,
                                           activation=None)

        self.add_norm = tft_add_and_norm_layer()

    def call(self, inputs, training):
        encoder_in, decoder_in, init_states = inputs

        encoder_out, enc_h, enc_c = self.tft_encoder(encoder_in, initial_state=init_states, training=training)
        
        decoder_out = self.tft_decoder(decoder_in, initial_state=[enc_h, enc_c], training=training)

        lstm_out = tf.concat([encoder_out, decoder_out], axis=1) # [batch, hist+future T, hidden_layer_size]

        # residual skip input
        lstm_in = tf.concat([encoder_in, decoder_in], axis=1)

        # Apply gating layer
        lstm_out, _ = self.gate(lstm_out, training=training)

        temporal_features = self.add_norm([lstm_out, lstm_in], training=training)

        return temporal_features


class static_enrichment_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, context, dropout_rate):
        super(static_enrichment_layer, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.context = context
        self.dropout_rate = dropout_rate
        self.grn_enrich = tft_grn_layer(hidden_layer_size=self.hidden_layer_size,
                                        output_size=None,
                                        dropout_rate=self.dropout_rate,
                                        use_time_distributed=True,
                                        additional_context=context,
                                        return_gate=False)

    def call(self,inputs, training):
        # inputs: [temporal_features, static_enrichment_vec]
        if self.context:
            x,c = inputs
            c = tf.expand_dims(c, axis=1)
            enriched = self.grn_enrich([x,c], training=training)
        else:
            x = inputs
            enriched = self.grn_enrich(x, training=training)

        return enriched


# Attention Layer

class Attention_Layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, n_head, dropout_rate):
        super(Attention_Layer, self).__init__()

        self.dropout_rate = dropout_rate
        self.hidden_layer_size = hidden_layer_size
        self.mha = TFTMultiHeadAttention(n_head=n_head,
                                         d_model=hidden_layer_size,
                                         dropout_rate=dropout_rate)

        self.gate = tft_apply_gating_layer(hidden_layer_size=self.hidden_layer_size,
                                           dropout_rate=self.dropout_rate,
                                           use_time_distributed=True,
                                           activation=None)
        self.add_norm = tft_add_and_norm_layer()

    def call(self, x, causal_mask, training):

        attn_out, _ = self.mha(x, x, x, causal_mask, training=training) # (q,k,v,mask,training)

        # gating layer
        attn_out, _ = self.gate(attn_out, training=training)
        # add_norm
        attn_out = self.add_norm([attn_out, x], training=training)
        return attn_out

class Attention_Stack(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_layer_size, n_head, dropout_rate):
        super(Attention_Stack, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.attn_layers = [Attention_Layer(hidden_layer_size, n_head, dropout_rate) for _ in range(num_layers)]
        self.grn_final = tft_grn_layer(hidden_layer_size = self.hidden_layer_size,
                                       output_size=None,
                                       dropout_rate=self.dropout_rate,
                                       use_time_distributed=True,
                                       additional_context=False,
                                       return_gate=False)

    def call(self, x, causal_mask, training):

        attn_out = x
        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](attn_out, causal_mask, training=training)

        # final GRN layer
        attn_out = self.grn_final(attn_out, training=training)
        return attn_out

class final_gating_layer(tf.keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate):
        super(final_gating_layer, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.gate = tft_apply_gating_layer(hidden_layer_size=self.hidden_layer_size,
                                           dropout_rate=self.dropout_rate,
                                           use_time_distributed=True,
                                           activation=None)
        self.add_norm = tft_add_and_norm_layer()

    def call(self, inputs, training):

        attn_out, temporal_features = inputs

        # final gating layer
        attn_out, _ = self.gate(attn_out, training=training)

        # final add & norm
        out = self.add_norm([attn_out, temporal_features], training=training)

        return out


# TFT Model

class TFT(tf.keras.Model):
    def __init__(self,
                 static_vars,
                 num_layers,
                 rnn_layers,
                 num_heads,
                 hidden_layer_size,
                 forecast_horizon,
                 max_inp_len,
                 loss_fn,
                 num_quantiles,
                 dropout_rate,
                 is_iqf):
        super(TFT, self).__init__()

        self.hist_len = max_inp_len
        self.f_len = forecast_horizon
        self.static_vars = static_vars
        self.hidden_layer_size = hidden_layer_size
        self.loss_fn = loss_fn
        self.rnn_layers = rnn_layers
        self.is_iqf = is_iqf
        self.num_quantiles = num_quantiles

        if self.static_vars:
            self.static_var_select_layer = variable_selection_static(hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate)
            self.static_context_layer = static_contexts(hidden_layer_size=hidden_layer_size, dropout_rate=dropout_rate)
            self.context = True

        else:
            self.context = False

        self.temporal_var_select_encoder_layer = variable_selection_temporal(hidden_layer_size=hidden_layer_size,
                                                                             static_context=self.context,
                                                                             dropout_rate=dropout_rate)

        self.temporal_var_select_decoder_layer = variable_selection_temporal(hidden_layer_size=hidden_layer_size,
                                                                             static_context=self.context,
                                                                             dropout_rate=dropout_rate)

        self.recurrent_layer = lstm_layer(hidden_layer_size=hidden_layer_size,
                                          rnn_layers=rnn_layers,
                                          dropout_rate=dropout_rate)

        self.self_attention = Attention_Stack(num_layers=num_layers,
                                              hidden_layer_size=hidden_layer_size,
                                              n_head=num_heads,
                                              dropout_rate=dropout_rate)

        self.static_enrich_layer = static_enrichment_layer(hidden_layer_size=hidden_layer_size,
                                                           context=self.context,
                                                           dropout_rate=dropout_rate)

        self.pff_layer = final_gating_layer(hidden_layer_size=hidden_layer_size,
                                            dropout_rate=dropout_rate)

        if self.loss_fn == 'Point':
            self.final_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_quantiles, activation=None))
        elif self.loss_fn == 'Binary':
            self.final_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        elif self.loss_fn == 'Poisson':
            self.final_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
        elif self.loss_fn == 'Quantile':
            if self.is_iqf:
                self.proj_intrcpt = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation=None))
                if self.num_quantiles > 1:
                    self.proj_incrmnt = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.num_quantiles-1, activation='softplus'))
            else:
                self.quantile_layers = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation=None)) 
                                        for _ in range(num_quantiles)]
        elif self.loss_fn == 'Normal':
            self.m_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1))
            self.s_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
        elif self.loss_fn == 'Negbin':
            self.m_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
            self.a_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='softplus'))
        else:
            raise ValueError('Specifiy supported loss function.')

    def call(self, inputs, training):
        """
        inputs: [list of static embedded vars, list of encoder embedded vars, list of decoder embedded vars, padding_mask, scale]

        """
        # variable selection & importance
        if self.context:
            stat_vars_list, encoder_vars_list, decoder_vars_list, padding_mask, scale = inputs

            static_vec, stat_weights = self.static_var_select_layer(stat_vars_list, training=training)
            context_vec, enrichment_vec, init_h, init_c = self.static_context_layer(static_vec, training=training)
            init_states = [init_h, init_c]

            encoder_inputs, past_weights = self.temporal_var_select_encoder_layer([encoder_vars_list, context_vec], training=training)
            decoder_inputs, future_weights = self.temporal_var_select_decoder_layer([decoder_vars_list, context_vec], training=training)
        else:
            encoder_vars_list, decoder_vars_list, padding_mask, scale = inputs
            batch_size = tf.shape(encoder_vars_list[0])[0]
            init_h = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            init_c = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            init_states = [init_h, init_c]

            encoder_inputs, past_weights = self.temporal_var_select_encoder_layer(encoder_vars_list, training=training)
            decoder_inputs, future_weights = self.temporal_var_select_decoder_layer(decoder_vars_list, training=training)

        # recurrent layer
        temporal_features = self.recurrent_layer([encoder_inputs, decoder_inputs, init_states], training=training)

        # static feature enrichment
        if self.context:
            enriched_features = self.static_enrich_layer([temporal_features, enrichment_vec], training=training)
        else:
            enriched_features = self.static_enrich_layer(temporal_features, training=training)

        # causal mask for decoder length
        mask = causal_mask(self.hist_len + self.f_len)

        # Attention stack
        attn_out = self.self_attention(enriched_features, mask, training=training)

        # pff
        attn_out = self.pff_layer([attn_out, temporal_features], training=training)

        attn_out = attn_out[:,-self.f_len:,:]

        # final output
        if self.loss_fn == 'Point':
            out = self.final_layer(attn_out, training=training)
        elif self.loss_fn == 'Binary':
            out = self.final_layer(attn_out, training=training)
        elif self.loss_fn == 'Poisson':
            mean = self.final_layer(attn_out, training=training)*scale
            parameters = mean
            out = poisson_sample(mu=mean)
        elif self.loss_fn == 'Normal':
            mean = self.m_layer(attn_out, training=training)*scale
            stddev = self.s_layer(attn_out, training=training)*scale
            parameters = tf.concat([mean, stddev], axis=-1)
            out = normal_sample(mean, stddev)
        elif self.loss_fn == 'Negbin':
            mean = self.m_layer(attn_out, training=training)*scale
            alpha = self.a_layer(attn_out, training=training)*tf.sqrt(scale)
            parameters = tf.concat([mean, alpha], axis=-1)
            out = negbin_sample(mean, alpha)
        elif self.loss_fn == 'Quantile':
            if self.is_iqf:
                if self.num_quantiles > 1:
                    out = tf.math.cumsum(tf.concat([self.proj_intrcpt(attn_out), self.proj_incrmnt(attn_out)], axis=-1), axis=-1)
                else:
                    out = self.proj_intrcpt(attn_out)
            else:
                output = []
                for i in range(len(self.quantile_layers)):
                    out = self.quantile_layers[i](attn_out, training=training)
                    output.append(out)
                out = tf.concat(output, axis=-1)

        if self.loss_fn == 'Point' or self.loss_fn == 'Binary' or self.loss_fn == 'Quantile':
            return out, scale, stat_weights, past_weights, future_weights
        else:
            return out, parameters, stat_weights, past_weights, future_weights
            


# TFT Wrapper

class TFT_Model(tf.keras.Model):
    def __init__(self,
                 col_index_dict,
                 vocab_dict,
                 num_layers,
                 num_heads,
                 d_model,
                 forecast_horizon,
                 max_inp_len,
                 loss_type,
                 num_quantiles=1,
                 dropout_rate=0.1):

        super(TFT_Model, self).__init__()
        
        self.hist_len = int(max_inp_len)
        self.f_len = int(forecast_horizon)
        self.loss_type = loss_type
        self.col_index_dict = col_index_dict
        self.vocab_dict = vocab_dict
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.max_inp_len = max_inp_len
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        
        if (len(self.col_index_dict.get('static_num_indices')[0])==0) and (len(self.col_index_dict.get('static_cat_indices')[0])==0):
            self.static_variables = False
        else:
            self.static_variables = True
        
        self.model = TFT(static_vars = self.static_variables,
                         num_layers = self.num_layers,
                         rnn_layers = 1,
                         hidden_layer_size = self.d_model,
                         num_heads = self.num_heads,       
                         forecast_horizon = self.f_len,
                         max_inp_len = self.hist_len, 
                         loss_fn = self.loss_type,
                         num_quantiles = self.num_quantiles,
                         dropout_rate = self.dropout_rate,
                         is_iqf = True)
        
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
        
        # Create Numerical Embedding (Linear Transform) Layers
        
        self.target_linear_transform_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=d_model, use_bias=False)) 
        self.scale_linear_transform_layer = tf.keras.layers.Dense(units=d_model, use_bias=False)
        
        if len(self.stat_num_col_names)>0:
            self.stat_linear_transform_layers = {}
            for colname in self.stat_num_col_names:
                self.stat_linear_transform_layers[colname] = tf.keras.layers.Dense(units=d_model, use_bias=False)
        
        if len(self.known_num_col_names)>0:
            self.known_linear_transform_layers = {}
            for colname in self.known_num_col_names + ['rel_age']:
                self.known_linear_transform_layers[colname] = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=d_model, use_bias=False))
        
        if len(self.unknown_num_col_names)>0:
            self.unknown_linear_transform_layers = {}
            for colname in self.unknown_num_col_names + ['rel_age']:
                self.unknown_linear_transform_layers[colname] = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=d_model, use_bias=False))
                
    def call(self, inputs, training):
        
        # target
        target = tf.strings.to_number(inputs[:,:,self.target_index:self.target_index+1], out_type=tf.dtypes.float32)
        target = self.target_linear_transform_layer(target)
        
        # ordered col names list
        stat_cols_ordered_list = []
        past_cols_ordered_list = [self.target_col_name]
        future_cols_ordered_list = []
        
        # stat, encoder, decoder tensor lists
        static_vars_list = []
        encoder_vars_list = [target[:,:self.hist_len,:]]
        decoder_vars_list = []
        
        # static numeric
        if len(self.stat_num_indices)>0:
            stat_cols_ordered_list = stat_cols_ordered_list + self.stat_num_col_names
            for col, i in zip(self.stat_num_col_names, self.stat_num_indices):
                stat_var = tf.strings.to_number(inputs[:,-1,i:i+1], out_type=tf.dtypes.float32)
                stat_var = self.stat_linear_transform_layers[col](stat_var)
                # append
                static_vars_list.append(stat_var)
        
        # static embeddings
        if len(self.stat_cat_indices)>0:
            stat_cols_ordered_list = stat_cols_ordered_list + self.stat_cat_col_names
            for col, i in zip(self.stat_cat_col_names, self.stat_cat_indices):
                stat_var = inputs[:,:,i]
                stat_var_id = self.stat_lookup_tables.get(col).lookup(stat_var)
                stat_var_embeddings = self.stat_embed_layers.get(col)(stat_var_id)
                # append
                static_vars_list.append(stat_var_embeddings[:,-1,:])
                
        # known numeric 
        if len(self.known_num_indices)>0:
            past_cols_ordered_list = past_cols_ordered_list + self.known_num_col_names
            future_cols_ordered_list = future_cols_ordered_list + self.known_num_col_names
            for col, i in zip(self.known_num_col_names, self.known_num_indices):
                num_vars = tf.strings.to_number(inputs[:,:,i:i+1], out_type=tf.dtypes.float32)
                num_vars = self.known_linear_transform_layers[col](num_vars)
                # append
                encoder_vars_list.append(num_vars[:,:self.hist_len,:])
                decoder_vars_list.append(num_vars[:,-self.f_len:,:])
        
        # unknown numeric
        if len(self.unknown_num_indices)>0:
            past_cols_ordered_list = past_cols_ordered_list + self.unknown_num_col_names
            for col, i in zip(self.unknown_num_col_names, self.unknown_num_indices):
                num_vars = tf.strings.to_number(inputs[:,:,i:i+1], out_type=tf.dtypes.float32)
                num_vars = self.unknown_linear_transform_layers[col](num_vars)
                # append
                encoder_vars_list.append(num_vars[:,:self.hist_len,:])
                     
        # known embeddings
        if len(self.known_cat_indices)>0:
            past_cols_ordered_list = past_cols_ordered_list + self.known_cat_col_names
            future_cols_ordered_list = future_cols_ordered_list + self.known_cat_col_names
            for col, i in zip(self.known_cat_col_names, self.known_cat_indices):
                cat_var = inputs[:,:,i]
                cat_var_id = self.temporal_known_lookup_tables.get(col).lookup(cat_var)
                cat_var_embeddings = self.temporal_known_embed_layers.get(col)(cat_var_id)
                # append
                encoder_vars_list.append(cat_var_embeddings[:,:self.hist_len,:])
                decoder_vars_list.append(cat_var_embeddings[:,-self.f_len:,:])
        
        # unknown embeddings
        if len(self.unknown_cat_indices)>0:
            past_cols_ordered_list = past_cols_ordered_list + self.unknown_cat_col_names
            for col, i in zip(self.unknown_cat_col_names, self.unknown_cat_indices):
                cat_var = inputs[:,:,i]
                cat_var_id = self.temporal_unknown_lookup_tables.get(col).lookup(cat_var)
                cat_var_embeddings = self.temporal_unknown_embed_layers.get(col)(cat_var_id)
                # append
                encoder_vars_list.append(cat_var_embeddings[:,:self.hist_len,:])
                          
        # rel_age
        rel_age = tf.strings.to_number(inputs[:,:,-3:-2], out_type=tf.dtypes.float32)
        rel_age_enc = self.known_linear_transform_layers['rel_age'](rel_age[:,:self.hist_len,:])
        rel_age_dec = self.unknown_linear_transform_layers['rel_age'](rel_age[:,-self.f_len:,:])
        # append
        encoder_vars_list.append(rel_age_enc)
        decoder_vars_list.append(rel_age_dec)
        
        # scale
        scale = tf.strings.to_number(inputs[:,:,-2:-1], out_type=tf.dtypes.float32)
        scale_log = tf.math.log(tf.math.sqrt(scale))
        scale_log = self.scale_linear_transform_layer(scale_log[:,-1,:])
        # append
        static_vars_list.append(scale_log)
        
        # Append additional columns
        stat_cols_ordered_list = stat_cols_ordered_list + ['scale']
        past_cols_ordered_list = past_cols_ordered_list + ['rel_age']
        future_cols_ordered_list = future_cols_ordered_list + ['rel_age']
        
        # mask
        mask = tf.strings.to_number(inputs[:,:,-1:], out_type=tf.dtypes.float32)
        
        # model process
        if self.static_variables:
            o, s, s_wts, p_wts, f_wts = self.model([static_vars_list, encoder_vars_list, decoder_vars_list, mask, scale], training=training)
        else:
            o, s, s_wts, p_wts, f_wts = self.model([encoder_vars_list, decoder_vars_list, mask, scale], training=training)
        
        # Average feature weights across time dim
        p_wts = tf.math.reduce_mean(p_wts, axis=1)
        f_wts = tf.math.reduce_mean(f_wts, axis=1)
        
        return o, s, ([stat_cols_ordered_list,past_cols_ordered_list,future_cols_ordered_list], [s_wts, p_wts, f_wts])
    
    
def TFT_Train(model, 
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
            o, s, f = model(x_train, training=training)
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
        o, s, f = model(x_test, training=training)
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

    model_tracker_file.write('Feature Weighted Transformer Training started with following Model Parameters ... \n')
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
        
def TFT_Infer(model, inputs, loss_type, hist_len, f_len, target_index, num_quantiles):
    infer_tensor, scale, id_arr, date_arr = inputs
    scale = scale[:,-1:,-1]
    window_len = hist_len + f_len
    stat_wts_df = None 
    encoder_wts_df = None 
    decoder_wts_df = None
        
    out, dist, feature_wts = model(infer_tensor, training=False)
            
    if loss_type in ['Normal','Poisson','Negbin']:
        dist = dist.numpy()
        output_arr = dist[:,:,0]
        
    elif loss_type in ['Point']:
        out = out.numpy()
        output_arr = out[:,:,0]
    
    elif loss_type in ['Quantile']:
        out = out.numpy()
        output_arr = out[:,:,:]
             
    column_names_list, wts_list = feature_wts
    stat_columns, encoder_columns, decoder_columns = column_names_list
    stat_columns_string = []
    encoder_columns_string = []
    decoder_columns_string = []
    for col in stat_columns:
        stat_columns_string.append(col) 
    for col in encoder_columns:
        encoder_columns_string.append(col)
    for col in decoder_columns:
        decoder_columns_string.append(col)
                
    stat_wts, encoder_wts, decoder_wts = wts_list

    # Average feature weights across time dim
    encoder_wts = encoder_wts.numpy()
    decoder_wts = decoder_wts.numpy()
    
    # convert wts to df    
    encoder_wts_df = pd.DataFrame(encoder_wts, columns=encoder_columns_string)   
    decoder_wts_df = pd.DataFrame(decoder_wts, columns=decoder_columns_string)    
    if stat_wts is not None:
        stat_wts = stat_wts.numpy()
        stat_wts_df = pd.DataFrame(stat_wts, columns=stat_columns_string)   
            
    # rescale if necessary
    if loss_type in ['Normal','Poisson','Negbin']:
        output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1),output_arr), axis=1))
        output_df = output_df.melt(id_vars=0).sort_values(0).drop(columns=['variable']).rename(columns={0:'id','value':'forecast'})
        output_df = output_df.rename_axis('index').sort_values(by=['id','index']).reset_index(drop=True)
        
    elif loss_type in ['Point']:
        output_df = pd.DataFrame(np.concatenate((id_arr.reshape(-1,1),output_arr*scale.reshape(-1,1)), axis=1))
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
        
    # weights df merge with id
    stat_wts_df = pd.concat([pd.DataFrame(id_arr.reshape(-1,1)), stat_wts_df], axis=1)
    encoder_wts_df = pd.concat([pd.DataFrame(id_arr.reshape(-1,1)), encoder_wts_df], axis=1)
    decoder_wts_df = pd.concat([pd.DataFrame(id_arr.reshape(-1,1)), decoder_wts_df], axis=1)
    print(stat_wts_df.shape, encoder_wts_df.shape, decoder_wts_df.shape)
        
    return forecast_df, stat_wts_df, encoder_wts_df, decoder_wts_df


class Temporal_Fusion_Transformer:
    def __init__(self, 
                 col_index_dict,
                 vocab_dict,
                 num_layers,
                 num_heads,
                 d_model,
                 forecast_horizon,
                 max_inp_len,
                 loss_type,
                 num_quantiles=1,
                 dropout_rate=0.1):
        
        self.col_index_dict = col_index_dict
        self.vocab_dict = vocab_dict
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.max_inp_len = max_inp_len
        self.loss_type = loss_type
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        self.target_col_name, self.target_index = self.col_index_dict.get('target_index')
    
    def build(self):
        tf.keras.backend.clear_session()
        self.model = TFT_Model(self.col_index_dict,
                                  self.vocab_dict,
                                  self.num_layers,
                                  self.num_heads,
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
        
        best_model = TFT_Train(self.model, 
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
        forecast, stat_wts_df, encoder_wts_df, decoder_wts_df = TFT_Infer(self.model, inputs, self.loss_type, self.max_inp_len, self.forecast_horizon, self.target_index, self.num_quantiles)
        
        return forecast, [stat_wts_df, encoder_wts_df, decoder_wts_df]
       
