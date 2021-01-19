import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import (Conv2D, 
  GlobalAveragePooling2D, 
  Dense, 
  Dropout,
  BatchNormalization
)
from tensorflow.keras.layers.experimental.preprocessing import (
  RandomCrop,
  RandomRotation
)

from micron2.clustering.utils import perturb_x

"""
The set encoder model must accept a batch of sets (5D):

x ~ [batch_size, set_size, h, w, ch]

We'll go for a nested model structure with an InnerModel
that takes one of the sets that looks like a normal batch (4D):

x_inner ~ [set_size, h, w, ch]
"""

class BasicCNN(tf.keras.Model):
  def __init__(self):
    self.conv1 = Conv2D(64, 5, activation='relu', name='conv1')
    self.bn1 = BatchNormalization(name='BN1')
    self.conv2 = Conv2D(128, 3, activation='relu', name='conv2')
    self.bn2 = BatchNormalization(name='BN2')
    self.conv3 = Conv2D(256, 3, activation='relu', name='conv3')
    self.bn3 = BatchNormalization(name='BN3')
    self.pool = GlobalAveragePooling2D(name='GAP2D')

  def call(self, x, trainint=True):
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.conv3(x)
    x = self.bn3(x, training=training)
    x = self.pool(x)
    return x




class InnerModel(tf.keras.Model):
  def __init__(self, inner_dim=128, crop_size=48, n_channels=3, encoder_type='basic_cnn'):
    super(InnerModel, self).__init__()
    self.inner_dim = inner_dim

    # self.pp_rotate = RandomRotation(factor=(0.5, 0.5), fill_mode='constant',
    #                                 fill_value=0, name='pp_rotate')
    # self.pp_crop = RandomCrop(crop_size, crop_size, name='pp_crop')

    if encoder_type == 'basic_cnn':
      self.encoder = BasicCNN()
    elif encoder_type == 'keras_resnet':
      self.encoder = tf.keras.applications.ResNet50V2(include_top=False, weights=None,
                                                      input_shape=(crop_size, crop_size, n_channels),
                                                      pooling='avg')

    self.crop_size = crop_size
    self.out = Dense(inner_dim, activation='relu', name='dense')

  def call(self, x, training=True):
    # x = self.pp_rotate(x, training=training)
    # x = self.pp_crop(x, training=training)

    # TODO switch off and replace with center crop for training=False
    x = perturb_x(x, self.crop_size)

    x = self.encoder(x)
    x = self.out(x)
    return x




class gModel(tf.keras.Model):
  def __init__(self, g_dim=32):
    super(gModel, self).__init__()
    self.dense1 = Dense(g_dim * 4, activation = 'relu')
    self.dense2 = Dense(g_dim * 2, activation = 'relu')
    # No activation on the output -- but regularize it ( g gets regularized in the loss function )
    self.projection = Dense(g_dim, activation=None) 
  
  def call(self, x):
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.projection(x)
    return x




class SetEncoding(tf.keras.Model):
  def __init__(self, inner_dim=128, outer_dim=64, g_dim=32, crop_size=48, 
               n_channels=3,
               encoder_type='basic_cnn'):
    super(SetEncoding, self).__init__()
    self.inner_model = InnerModel(inner_dim=inner_dim, 
                                  crop_size=crop_size, 
                                  n_channels=n_channels, 
                                  encoder_type=encoder_type)
    self.out_layer = Dense(outer_dim, activation=None, name='OUTER_dense')

    self.g_fn = gModel(g_dim=g_dim)

  def flatten_input(self, x):
    stack_size = x.shape[0] * x.shape[1]
    input_batch_size = x.shape[0]
    stack_shape = [stack_size, x.shape[2], x.shape[3], x.shape[4]]
    x = tf.reshape(x, stack_shape)
    return x, input_batch_size


  def model_backbone(self, x, training=True):
    # we can reshape then shape back
    # might need a special kernel to keep everything in line
    # we want the result: x ~ [batch_size * set_size, h, w, c]
    x, input_batch_size = self.flatten_input(x)

    # process it to x ~ [batch_size * set_size, z]
    x = self.inner_model(x, training=training)

    # then we want to restore it to: x ~ [batch_size , set_size, z]
    x = tf.stack( tf.split(x, input_batch_size, axis=0), axis=0 )

    # Apply summary function -- mean, sum, max, attention
    x = tf.reduce_mean(x, axis=1)
    x = self.out_layer(x) # And one more layer

    return x

  def encode(self, x, training=True):
    x = self.model_backbone(x, training=training)
    # regularize the set representation
    x = tf.math.l2_normalize(x, axis=1)
    return x

  def encode_g(self, x, training=True):
    x = self.encode(x)
    g = self.g_fn(x)
    return g

  def call(self, x, return_g=False, training=True):
    x = self.encode(x, training=training)
    if return_g:
      g = self.g_fn(x)
      return x, g
    else:
      return x


if __name__ == '__main__':
  # for new GPUs (in the 3xxx series) we need these two lines
  # last tested with:
  # tensorflow==2.4.0 (via pip)
  # Driver Version: 455.38 (system)
  # (via conda):
  # cudatoolkit               11.0.221             h6bb024c_0    <unknown>
  # cudnn                     8.0.4                cuda11.0_0    nvidia
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  print('testing set encoding model')
  model = SetEncoding(crop_size=24)

  x = np.zeros((32, 8, 28, 28, 1), dtype=np.float32)
  print('Input:', x.shape, x.dtype)

  z = model(x)
  print('Model returned', z.shape)

  for v in model.trainable_variables:
    print(f'{v.name:<30}{v.shape}')

