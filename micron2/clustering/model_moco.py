import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tqdm.auto as tqdm

from tensorflow.keras.callbacks import Callback
import gc

from tensorflow.keras.layers import (Dense, 
  Conv2D, Dropout, BatchNormalization, 
  Conv2DTranspose,
  Flatten
)

# from tensorflow.keras.layers.experimental.preprocessing import (
#   RandomFlip,
#   RandomRotation,
#   RandomTranslation,
#   RandomCrop,
#   RandomContrast,
#   # Rescaling
# )

# https://github.com/PaperCodeReview/MoCo-TF/blob/master/callback.py
import sys
import gc


class UpdateQueue(Callback):
  def __init__(self, momentum, max_queue_len):
    super(UpdateQueue, self).__init__()
    self.max_queue_len = max_queue_len
    self.momentum = momentum

  def on_batch_end(self, batch, logs=None):
    for mv, kv in zip(self.model.encode_g.trainable_variables, self.model.encode_k.trainable_variables):
      tf.compat.v1.assign(kv, self.momentum * kv + (1-self.momentum) * mv )

    key = logs.pop('key')
    # self.model.queue = tf.concat([tf.transpose(key), self.model.queue], axis=-1)
    self.model.Q = tf.concat([self.model.Q, key], axis=0)
    self.model.Q = self.model.Q[:self.max_queue_len, :]
    tf.keras.backend.clear_session()
    # del(key)
    # gc.collect()


def _get_encoder(encoder_type, input_shape):
  app_args = dict(include_top=False, weights=None,
                  input_shape=input_shape,
                  pooling='average')
  if encoder_type == 'ResNet50V2':
    return tf.keras.applications.ResNet50V2(**app_args)
  elif encoder_type == 'EfficientNetB1':
    return tf.keras.applications.EfficientNetB1(**app_args)
  else:
    # Default
    return tf.keras.applications.ResNet50V2(**app_args)


class Classifier(tf.keras.Model):
  def __init__(self, encoder, n_classes=None, mlp_dim=128):
    """ Encode an image into a reduced 1D representation """
    super(Classifier, self).__init__()
    assert n_classes is not None
    self.encoder = encoder
    self.n_classes = n_classes
    self.mlp_dim = mlp_dim

    self.mlp_hid = Dense(self.mlp_dim, activation='relu')
    self.mlp_out = Dense(self.n_classes, activation='softmax')

  def call(self, x, training=True):
    z = self.encoder(x, training=training)
    z = self.mlp_hid(z)
    y = self.mlp_out(z)
    return y


class Encoder(tf.keras.Model):
  def __init__(self, data_shape=[64, 64, 3], z_dim=256, 
               encoder_type='ResNet50V2'):
    """ Encode an image into a reduced 1D representation """
    super(Encoder, self).__init__()

    self.x_size = data_shape[0]

    self.backbone = _get_encoder(encoder_type, data_shape)
    self.conv_2 = Conv2D(filters=512, kernel_size=(2,2), strides=(1,1), 
               padding='same', activation='relu')
    self.flat = Flatten()
    self.dense_1 = Dense(512, activation='relu')
    self.dense_2 = Dense(z_dim, activation=None)

  def call(self, x, training=True):
    # if training:
    #   x = self.perturb(x)
    x = self.backbone(x, training=training)
    x = self.conv_2(x)
    x = self.flat(x)
    x = self.dense_1(x)
    x = self.dense_2(x)
    x = tf.math.l2_normalize(x, axis=1)
    return x


class MoCo(tf.keras.Model):
  def __init__(self, data_shape=[64, 64, 3], z_dim=256, max_queue_len=4096, batch_size=64,
               momentum=0.999, temp=1.0, encoder_type='ResNet50V2'):
    """ Encode an image into a reduced 1D representation """
    super(MoCo, self).__init__()
    self.n_channels = data_shape[-1]
    self.z_dim = z_dim
    self.encoder_type = encoder_type
    self.data_shape = data_shape
    self.momentum = momentum
    self.temp = temp
    self.batch_size = batch_size
    self.max_queue_len = max_queue_len

    self._Q = np.random.normal(size=(self.max_queue_len, self.z_dim)).T
    self._Q /= np.linalg.norm(self._Q, axis=0)
    self._Q = self._Q.T
    print(f'Q: {self._Q.shape}')
    self.Q = self.add_weight(name='queue', 
                             shape=(self.max_queue_len, self.z_dim),
                             initializer=tf.keras.initializers.Constant(self._Q),
                             trainable=False)
    
    self.encode_g = Encoder(data_shape=data_shape, z_dim=z_dim, 
                            encoder_type=encoder_type)
    self.encode_k = Encoder(data_shape=data_shape, z_dim=z_dim,
                            encoder_type=encoder_type)


  def call(self, x):
    x = self.encode_g(x)
    x = tf.math.l2_normalize(x, axis=1)
    return x


  def compile(self, optimizer, loss, metrics=None, num_workers=1, run_eagerly=None):
    super(MoCo, self).compile(optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)
    self._loss = loss


  def enqueue(self, z):
    # n = z.shape[0]
    i = tf.range(self.max_queue_len) < self.batch_size
    q = self.Q[~i]
    self.Q = tf.concat([q, z], axis=0)


  def update_key_model(self):
    """ 
    the order of variables in model and kmodel should be identical
    """
    for mv, kv in zip(self.encode_g.trainable_variables, self.encode_k.trainable_variables):
      tf.compat.v1.assign(kv, self.momentum * kv + (1-self.momentum) * mv )


  def moco_loss(self, q_feat, key_feat):
    """ 
    q_feat: NxC
    key_feat: NxC
    queue: KxC (all elements of the queue)
    batch_size: int
    temp: float
    """
    ## https://github.com/ppwwyyxx/moco.tensorflow/blob/master/main_moco.py
    # loss
    l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))  # nx1
    l_neg = tf.einsum('nc,kc->nk', q_feat, self.Q)  # nxK
    logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
    logits = logits * (1 / self.temp)
    labels = tf.zeros(self.batch_size, dtype=tf.int64)  # n
    # labels = tf.zeros_like(logits, dtype=tf.int64)  # n
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss, name='xentropy-loss')
    return loss
  

  def train_step(self, batch):
    key_feat = self.encode_k(batch)
    # key_feat = tf.math.l2_normalize(key_feat, axis=1)

    with tf.GradientTape() as tape:
      q = self.encode_g(batch)
      # q = tf.math.l2_normalize(q, axis=1)
      loss = self.moco_loss(q, tf.stop_gradient(key_feat))#, batch_size=batch.shape[0])

    variables = self.encode_g.trainable_variables
    grads = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(grads, variables))
    gc.collect()

    results = {'key': key_feat, 'loss': loss}
    return results




class MoCo_Classifier(MoCo):
  def __init__(self, data_shape=[64, 64, 3], z_dim=256, max_queue_len=4096, 
               n_classes=None, mlp_dim=128, batch_size=64, momentum=0.999, temp=1.0,
               alpha=0.01, beta=0.99,
               encoder_type='ResNet50V2'):

    """ Encode an image into a reduced 1D representation """
    super(MoCo_Classifier, self).__init__(data_shape=data_shape, z_dim=z_dim, max_queue_len=max_queue_len, 
                                          batch_size=batch_size, momentum=momentum, temp=temp, 
                                          encoder_type=encoder_type)

    assert n_classes is not None
    self.classifier = Classifier(encoder=self.encode_g, n_classes=n_classes, mlp_dim=mlp_dim)
    self.xent = tf.keras.losses.CategoricalCrossentropy()
    self.alpha = alpha
    self.beta = beta

  def train_step(self, batch):
    batch_images, batch_labels = batch
    key_feat = self.encode_k(batch_images)
    # key_feat = tf.math.l2_normalize(key_feat, axis=1)

    sample_weight = tf.reduce_sum(batch_labels, axis=1) 
    sample_weight = sample_weight / (tf.reduce_sum(sample_weight) + 1e-7)

    with tf.GradientTape() as tape:
      q = self.encode_g(batch_images)
      # q = tf.math.l2_normalize(q, axis=1)

      # skip processing with G again
      y = self.classifier.mlp_hid(q)
      y = self.classifier.mlp_out(y)
      xe = self.xent(batch_labels, y, sample_weight=sample_weight)

      # loss = self.moco_loss(q, tf.stop_gradient(key_feat))#, batch_size=batch.shape[0])
      loss = self.alpha * xe + self.beta * self.moco_loss(q, tf.stop_gradient(key_feat))

    variables = self.encode_g.trainable_variables + self.classifier.trainable_variables
    grads = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(grads, variables))
    gc.collect()

    results = {'key': key_feat, 'loss': loss}
    return results