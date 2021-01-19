import os
import numpy as np
import tensorflow as tf

import tqdm.auto as tqdm

# from micron2.data import stream_dataset
from micron2.clustering.embedding_simclr import simclr_loss_fn
from .set_model import SetEncoding
from .set_model import gModel


def train_sets_SimCLR(dataset, model, lr=1e-4, tau=0.1):
  """
  Set up a dataset with arbitrary preprocessing like this:
  from micron2 import stream_dataset

  def process(x):
    x = tf.cast(x, tf.float32)/255.
    # whatever else you want to do
    return x
    
  dataset = stream_dataset('setdataset.hdf5', use_channels=['DAPI', 'CD45', 'PanCytoK'],)
  dataset = (dataset.repeat(10)
            .shuffle(512)
            .map(process)
            .batch(16)
            .prefetch(8)
            #.apply(tf.data.experimental.prefetch_to_device("/gpu:0")))

  # Define a model ....
  model = SetEmbedding()

  # Then call this function 
  loss_history = train_sets_SimCLR(dataset, model)
  
  """

  optim = tf.keras.optimizers.Adam(learning_rate=lr)
  pbar = tqdm.tqdm(enumerate(dataset))

  loss_history = []
  losses = []
  for i, batch in pbar:
    batch = tf.tile(batch, (2, 1, 1, 1, 1))
    with tf.GradientTape() as tape:
      g = model.encode_g(batch, training=True)
      l_simclr = simclr_loss_fn(g, tau=tau)

    simclr_grads = tape.gradient(l_simclr, model.trainable_variables)
    optim.apply_gradients(zip(simclr_grads, model.trainable_variables))

    l = l_simclr.numpy()
    loss_history.append(l)
    losses.append(l)
    if i % 100 == 0:
      mn = np.mean(losses)
      s = f'loss={l:<3.4e}\tMN_LOSS={mn:<3.4e}'
      pbar.set_description(s)

    if i % 1000 == 0:
      losses = []

  return loss_history