import os
import numpy as np
import pandas as pd

import tensorflow as tf
import tqdm.auto as tqdm

from tensorflow.keras.layers import (Dense, 
  Conv2D, Dropout, BatchNormalization, Conv2DTranspose,
  Flatten
  )

from tensorflow.keras.layers.experimental.preprocessing import (
  RandomFlip,
  RandomRotation,
  RandomTranslation,
  RandomCrop,
  RandomContrast
)


class Encoder(tf.keras.Model):
  def __init__(self, input_shape=[64, 64, 3], z_dim=128, g_network=True):
    """ Encode an image into a reduced 1D representation """
    super(Encoder, self).__init__()
    self.n_channels = input_shape[-1]
    self.x_size = input_shape[0]
    self.z_dim = z_dim
    
    # perturb functions
    # self.contrast = RandomContrast()
    self.flip = RandomFlip()
    self.rotate = RandomRotation(1, fill_mode='constant', fill_value=0)
    self.translate = RandomTranslation(height_factor=(0.1, 0.1), 
                                       width_factor=(0.1,0.1),
                                       fill_mode='constant',
                                       fill_value=0)
    self.crop = RandomCrop(self.x_size, self.x_size)


    self.conv_1 = tf.keras.applications.ResNet50V2(include_top=False, weights=None,
                             input_shape=input_shape,
                             pooling='average')

    self.conv_2 = Conv2D(filters=512, kernel_size=(2,2), strides=(1,1), 
               padding='same', activation='relu')
    self.flat = Flatten()

    self.dense_1 = Dense(512, activation='relu')
    self.dense_2 = Dense(self.z_dim, activation=None)

    self.g_network = g_network
    if self.g_network:
      self.g_fn_0 = Dense(128, activation='relu')
      self.g_fn_1 = Dense(32, activation=None)

  def perturb(self, x):
    x = self.flip(x)
    x = self.rotate(x)
    x = self.translate(x)
    x = self.crop(x)
    return x

  def model_backbone(self, x, training=True):
    if training:
      x = self.perturb(x)

    x = self.conv_1(x, training=training)
    x = self.conv_2(x)
    x = self.flat(x)
    x = self.dense_1(x)
    x = self.dense_2(x)
    return x

  def call(self, x, return_g=False, training=True):
    x = self.model_backbone(x, training=training)

    if return_g:
      g = self.g_fn_0(x)
      g = self.g_fn_1(g)
      return x, g
    else:
      return x

  def encode(self, x, training=True):
    x = self.model_backbone(x, training=training)
    x = tf.math.l2_normalize(x, axis=1)
    return x

  def encode_g(self, x, training=True):
    # x = self.model_backbone(x, training=training)
    # x = tf.math.l2_normalize(x, axis=1)
    x = self.encode(x, training=training)
    g = self.g_fn_0(x)
    g = self.g_fn_1(g)
    return g




class Autoencoder(tf.keras.Model):
  def __init__(self, input_shape=[64, 64, 3], g_network=True):
    super(Autoencoder, self).__init__()
    self.n_channels = input_shape[-1]
    self.n_upsamples = 7
    self.conv_1 = tf.keras.applications.ResNet50V2(include_top=False, weights=None,
                             input_shape=input_shape,
                             pooling='average')
    self.conv_2 = Conv2D(filters=256, kernel_size=(2,2), strides=(1,1), 
               padding='same', activation='relu')
    
    self.g_network = g_network
    if self.g_network:
      self.g_fn = Dense(32, activation=None, name='g_simclr')
    
    self.build_upsample(name='up_large')
    self.build_upsample(name='up_small')
    
  def build_upsample(self, name='upsample'):
    p_act = dict(padding='same', activation='relu')
    setattr(self, f'{name}_0', Conv2DTranspose(filters=256,  kernel_size=(2,2),
                    strides=(2,2), **p_act))
    setattr(self, f'{name}_1', Conv2DTranspose(filters=128,  kernel_size=(3,3),
                    strides=(2,2), **p_act))
    setattr(self, f'{name}_2', Conv2DTranspose(filters=64,  kernel_size=(5,5),
                    strides=(2,2), **p_act))
    setattr(self, f'{name}_3', Conv2DTranspose(filters=64,  kernel_size=(5,5),
                    strides=(2,2), **p_act))
    setattr(self, f'{name}_4', Conv2DTranspose(filters=64,  kernel_size=(3,3),
                    strides=(1,1), **p_act))
    setattr(self, f'{name}_5', Conv2DTranspose(filters=self.n_channels,  kernel_size=(5,5),
                    strides=(2,2), **p_act))
    setattr(self, f'{name}_6', Conv2DTranspose(filters=self.n_channels,  kernel_size=(3,3),
                    strides=(1,1), padding='same', activation=None))
    
  def apply_upsample(self, z, name='upsample'):
    for j in range(self.n_upsamples):
      z = getattr(self, f'{name}_{j}')(z)
    return z
    
  def call(self, x, return_g=False):
    x1 = self.conv_1(x)
    x2 = self.conv_2(x1)
    
    # Two parallel upsampling paths
    x1up = self.apply_upsample(x1, name='up_large')
    x2up = self.apply_upsample(x2, name='up_small')

    xout = tf.reduce_mean([x1up, x2up], axis=0)
    xout = tf.image.resize_with_crop_or_pad(xout, x.shape[1], x.shape[2])
    if return_g:
      g = self.g_fn(tf.reduce_mean(x2, axis=[1,2]))
      return xout, g
    else:
      return xout  
  
  def encode_g(self, x):
    if not self.g_network:
      raise ValueError('Network not instantiated with a G_fn. Use Autoencoder(g_network=True)')

    # Apply g function for simclr
    x = self.conv_1(x)
    x = self.conv_2(x)
    x = tf.reduce_mean(x, axis=[1,2])
    x = self.g_fn(x)
    return x
  
  def encode(self, x, retval=1):
    if retval == 0:
      x = self.conv_1(x)
    elif retval == 1:
      x = self.conv_1(x)
      x = self.conv_2(x)
      
    return tf.reduce_mean(x, axis=[1,2])