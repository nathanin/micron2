import os
import numpy as np
import pandas as pd

import tensorflow as tf
# from micron2.data import stream_dataset

from .objective import add_contrastive_loss

import tqdm.auto as tqdm

from .utils import perturb_x

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
  

  
  
# Autoencoder + simCLR. Use gradient accumulation for simCLR loss. 
# Autoencoder is trained every step
def train_AE_simCLR(dataset, model, batch_reps=4, crop_size=48):
  """
  Set up a dataset with arbitrary preprocessing like this:
  from micron2.codexutils import stream_dataset

  def process(x):
    x = tf.cast(x, tf.float32)/255.
    return x
    
  dataset = stream_dataset('tests/dataset.hdf5', use_channels=['DAPI', 'CD45', 'PanCytoK'],)
  dataset = (dataset.repeat(10)
            .shuffle(1024 * 6)
            .map(process)
            .batch(16)
            .prefetch(8)
            #.apply(tf.data.experimental.prefetch_to_device("/gpu:0")))
  """
  mse_fn = tf.keras.losses.MeanSquaredError()
  optim = tf.keras.optimizers.Adam(learning_rate = 1e-3)
  simclr_optim = tf.keras.optimizers.Adam(learning_rate = 1e-4)
  pbar = tqdm.tqdm(enumerate(dataset))
  losses, sc_losses = [], []
  prev_loss, prev_sc_loss = 0, 0
  
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
  ae_vars = [v for v in model.trainable_variables if 'g_simclr' not in v.name]
  
  grad_dict = {v.name: [] for v in model.trainable_variables}
  for i, batch in pbar:
    # batch_in = tf.image.random_crop(batch, size=(batch.shape[0], crop_size, crop_size, batch.shape[-1]))
    batch = tf.tile(batch, [batch_reps, 1, 1, 1])
    batch = tf.tile(batch, [2, 1, 1, 1])
    batch = perturb_x(batch, crop_size=crop_size)

    # batch_in = perturb_x(batch, crop_size=crop_size)
    # batch_p  = perturb_x(batch, crop_size=crop_size)
    # batch_in = tf.concat([batch_in, batch_p], axis=0)

    with tf.GradientTape(persistent=True) as tape:
      xout, g = model(batch, return_g=True)
      mse_loss = mse_fn(batch, xout)
      # l_simclr = simclr_loss_fn(g, tau=0.1)
      l_simclr, _, _ = add_contrastive_loss(g, temperature=0.1)
        
    losses.append(mse_loss.numpy())
    sc_losses.append(l_simclr.numpy())
    mse_grads = tape.gradient(mse_loss, ae_vars)
    simclr_grads = tape.gradient(l_simclr, model.trainable_variables)
    del tape
    
    optim.apply_gradients(zip(mse_grads, ae_vars))
    stash_grads(simclr_grads, grad_dict, model.trainable_variables)
    if i % 16 == 0:
      simclr_grads = mean_grads(grad_dict, model.trainable_variables)
      simclr_optim.apply_gradients(zip(simclr_grads, model.trainable_variables))
      grad_dict = {v.name: [] for v in model.trainable_variables}
        
    if i % 100 == 0:
      m_loss = np.mean(losses)
      m_sc_losses = np.mean(sc_losses)
      #pbar.set_description(f'mse_loss = {np.mean(losses):3.5e} simclr_loss = {np.mean(sc_losses):3.5e}')
      pbar.set_description(f'd(mse_loss) = {prev_loss-m_loss:3.5e}\t' +\
                           f'd(simclr_loss) = {prev_sc_loss-m_sc_losses:3.5e}')
      prev_loss = np.mean([prev_loss, m_loss])
      prev_sc_loss = np.mean([prev_sc_loss, m_sc_losses])
      losses, sc_losses = [], []
          

def train_simCLR(dataset, model, batch_reps=4, update_each=1, tau=0.1, 
                 lr=1e-3, crop_size=48):
  """
  Set up a dataset with arbitrary preprocessing like this:
  from micron2.codexutils import stream_dataset

  def process(x):
    x = tf.cast(x, tf.float32)/255.
    return x
    
  dataset = stream_dataset('tests/dataset.hdf5', use_channels=['DAPI', 'CD45', 'PanCytoK'],)
  dataset = (dataset.repeat(10)
            .shuffle(1024 * 6)
            .map(process)
            .batch(16)
            .prefetch(8)
            #.apply(tf.data.experimental.prefetch_to_device("/gpu:0")))
  """
  optim = tf.keras.optimizers.Adam(learning_rate = lr)
  pbar = tqdm.tqdm(enumerate(dataset))
  sc_losses = []
  prev_loss, prev_sc_loss = 0, 0
  
  def stash_grads(grads, grad_dict, trainable_variables):
    for i, v in enumerate(trainable_variables):
      if grads[i] is None:
        grad_dict[v.name].append(tf.zeros(v.shape, dtype=tf.float32))
      else:
        grad_dict[v.name].append(grads[i])
  
  def mean_grads(grad_dict, trainable_variables):
    grads = [tf.reduce_mean(grad_dict[v.name], axis=0) for v in trainable_variables]
    return grads
  
  loss_history = []
  grad_dict = {v.name: [] for v in model.trainable_variables}
  for i, batch in pbar:
    batch = tf.tile(batch, [batch_reps, 1, 1, 1]) # This is recommended for IIC ? 
    batch = tf.tile(batch, [2, 1, 1, 1]) # This is needed for simCLR
    batch = perturb_x(batch, crop_size=crop_size)

    with tf.GradientTape() as tape:
      g = model.encode_g(batch)
      # l_simclr = simclr_loss_fn(g, tau=tau)
      l_simclr, _, _ = add_contrastive_loss(g, temperature=0.1)
        
    l_simclr_np = l_simclr.numpy()
    sc_losses.append(l_simclr_np)
    loss_history.append(l_simclr_np)
    simclr_grads = tape.gradient(l_simclr, model.trainable_variables)
    
    stash_grads(simclr_grads, grad_dict, model.trainable_variables)
    if i % update_each == 0:
      simclr_grads = mean_grads(grad_dict, model.trainable_variables)
      optim.apply_gradients(zip(simclr_grads, model.trainable_variables))
      grad_dict = {v.name: [] for v in model.trainable_variables}
        
    if i % 100 == 0:
      m_sc_losses = np.mean(sc_losses)
      pbar.set_description(f'simclr_loss = {m_sc_losses:3.3e} \t d(simclr_loss) = {prev_sc_loss-m_sc_losses:3.3e}')
      prev_sc_loss = np.mean([prev_sc_loss, m_sc_losses])
      sc_losses = []

  return loss_history