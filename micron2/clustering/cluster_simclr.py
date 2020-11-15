import os
import numpy as np
import pandas as pd

import tensorflow as tf
from micron2.codexutils import stream_dataset

import tqdm.auto as tqdm

"""
Set up a dataset:

def process(x):
  x = tf.cast(x, tf.float32)/255.
  return x
  
use_channels=['DAPI', 'CD45', 'PanCytoK', 'CD31', 'PDGFRb', 'aSMA', 'Ki-67']
dataset = stream_dataset('dataset.hdf5', use_channels)
dataset = (dataset.repeat(10)
          .shuffle(1024 * 6)
          .map(process)
          .batch(8)
          .prefetch(1)
          )

"""

def similarity(u, tau=1.0):
  nu = tf.norm(u, ord=2, keepdims=True, axis=-1)
  sim = tf.tensordot(u, tf.transpose(u), 1) 
  return sim / (tf.constant(1e-9) + (tau * nu * tf.transpose(nu)))


def simclr_loss_fn(z_i, tau=1.):
  """
  A Simple Framework for Contrastive learning of Visual Representations
  Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, 2020.  
  https://arxiv.org/abs/2002.05709
  """

  s = tf.exp(similarity(z_i, tau=tau))
  i_part, j_part = tf.split(s, 2, 0)

  total_i = tf.reduce_sum(i_part, axis=-1) - tf.linalg.diag_part(i_part)
  total_j = tf.reduce_sum(j_part, axis=-1) - tf.linalg.diag_part(j_part)

  l_i = -tf.math.log( tf.linalg.diag_part(i_part) / total_i )
  l_j = -tf.math.log( tf.linalg.diag_part(j_part) / total_j )

  loss = tf.reduce_sum(l_i + l_j)
  return loss
  

def perturb_x(x):
  x = tf.image.random_flip_left_right(x)
  x = tf.image.random_flip_up_down(x)
  return x
  
  
# Autoencoder + simCLR. Use gradient accumulation for simCLR loss. 
# Autoencoder is trained every step
def train_loop_simCLR(dataset, model):
  mse_fn = tf.keras.losses.MeanSquaredError()
  optim = tf.keras.optimizers.Adam(learning_rate = 1e-4)
  simclr_optim = tf.keras.optimizers.Adam(learning_rate = 1e-4)
  pbar = tqdm.tqdm(enumerate(dataset))
  losses, sc_losses = [], []
  
  def stash_grads(grads, grad_dict, trainable_variables):
    for i, v in enumerate(trainable_variables):
      if grads[i] is None:
        grad_dict[v.name].append(tf.zeros(v.shape, dtype=tf.float32))
      else:
        grad_dict[v.name].append(grads[i])
  
  def mean_grads(grad_dict, trainable_variables):
    grads = [tf.reduce_mean(grad_dict[v.name], axis=0) for v in trainable_variables]
    return grads
  
  # Grab variables for the AE portion because otherwise theres tons of warnings
  ae_vars = [v for v in ae_model.trainable_variables if 'g_simclr' not in v.name]
  
  grad_dict = {v.name: [] for v in model.trainable_variables}
  for i, batch in pbar:
    with tf.GradientTape(persistent=True) as tape:
      xout = model(batch)
      mse_loss = mse_fn(batch, xout)
      g1 = model.encode_g(batch)
      
      batch = perturb_x(batch)
      g2 = model.encode_g(batch)
      l_simclr = simclr_loss_fn(tf.concat([g1, g2], axis=0))
      
    losses.append(mse_loss.numpy())
    sc_losses.append(l_simclr.numpy())
    mse_grads = tape.gradient(mse_loss, ae_vars)
    simclr_grads = tape.gradient(l_simclr, model.trainable_variables)
    del tape
    
    optim.apply_gradients(zip(mse_grads, ae_vars))
    stash_grads(simclr_grads, grad_dict, model.trainable_variables)
    if i % 32 == 0:
      simclr_grads = mean_grads(grad_dict, model.trainable_variables)
      simclr_optim.apply_gradients(zip(simclr_grads, model.trainable_variables))
      grad_dict = {v.name: [] for v in model.trainable_variables}
      
    if i % 100 == 0:
      pbar.set_description(f'mse_loss = {np.mean(losses):3.5e} simclr_loss = {np.mean(sc_losses):3.5e}')
      losses, sc_losses = [], []
      