import os
import numpy as np
import pandas as pd

import tensorflow as tf
from micron2.data import stream_dataset

import tqdm.auto as tqdm


# Normal training -- autoencoder only
def train_AE(dataset, model):
  mse_fn = tf.keras.losses.MeanSquaredError()
  optim = tf.keras.optimizers.Adam(learning_rate = 1e-4)

  pbar = tqdm.tqdm(enumerate(dataset))

  losses = []
  for i, batch in pbar:
    with tf.GradientTape() as tape:
      xout = model(batch)
      loss = mse_fn(batch, xout)
    losses.append(loss.numpy())
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    if i % 100 == 0:
      pbar.set_description(f'mean loss = {np.mean(losses):3.5e}')
      losses = []