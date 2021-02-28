import numpy as np
import tensorflow as tf
import tqdm.auto as tqdm

from .utils import perturb_x


# class MoCoQueue:
#   def __init__(self, max_queue_len=8, zsize=128):
#     # self.Q = []
#     self.max_queue_len = max_queue_len
#     self.zsize = zsize
#     self.Q = tf.zeros((max_queue_len, zsize), dtype=tf.float32)
  
#   def __len__(self):
#     return self.Q.shape[0]
      
#   def enqueue(self, z):
#     self.Q.append(z)
#     if len(self) > self.max_queue_len:
#       self.dequeue()
  
#   def dequeue(self):
#     """ Remove the oldest item in queue """
#     _ = self.Q.pop(0)
  
#   def getqueue(self):
#     return tf.concat(self.Q, axis=0)

class MoCoQueue:
    def __init__(self, max_queue_len=4096, zsize=128):
        self.Q = []
        self.max_queue_len = max_queue_len
        self.zsize = zsize
        with tf.device('/CPU:0'):
          self.Q = tf.zeros((max_queue_len, zsize), dtype=tf.float32)

    def __len__(self):
        return self.Q.shape[0]

    def enqueue(self, z):
        #self.Q.append(z)
        n = z.shape[0]
        i = tf.range(self.max_queue_len) < n
        q = self.Q[~i]
        self.Q = tf.concat([q, z], axis=0)
  

def moco_loss(q_feat, key_feat, queue, batch_size=1, temp=1.):
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
  l_neg = tf.einsum('nc,kc->nk', q_feat, queue)  # nxK
  logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
  logits = logits * (1 / temp)
  labels = tf.zeros(batch_size, dtype=tf.int64)  # n
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  loss = tf.reduce_mean(loss, name='xentropy-loss')
  return loss



def update_key_model(model, kmodel, momentum=0.999):
  """ 
  the order of variables in model and kmodel should be identical
  """
  for mv, kv in zip(model.trainable_variables, kmodel.trainable_variables):
    tf.compat.v1.assign(kv, momentum * kv + (1-momentum) * mv )




def train_moco(dataset, model, kmodel, max_queue_len=32, crop_size=48, 
               momentum=0.999, lr=1e-4, max_steps=1e5, temp=1.0, zsize=128,
               perturb=True, perturb_fn=perturb_x):
  """ Train an image embedding model with Momentum Contrast

  Momentum Contrast seeks to apply unsupervised contrastive learning by comparing a query batch against a bank of key batches. 
  Contrastive learning is thought to require a large number of negative examples to be effective.
  To meet this need, other methods process very large batches and are limited by GPU memory.
  MoCo solves this problem by using a queue of previously encoded batches as negative examples
  Each batch is processed by the main model, and an auxiliary ("key") model that is not backpropagated through. 
  The main model is updated by gradient descent with each training step as normal 
  The key model is slowly updated by taking a weighted average of its parameters, and the main model.
  Thus, small batch sizes coupled with a large queue can yield good results.

  Image perturbations are an important part of contrastive learning. 
  This function has options to apply random cropping and flipping by default.
  A customized perturb function can be provided, that should accept and return tensors.

  Args:
    model, kmodel (tf.keras.Model): identically shaped tf.keras.Model instances, with two sets of parameters
    max_queue_len (int): number of batches to hold in the key queue
    crop_size (int) : for live perturbations
    momentum (float): control the update of the key encoder (usually close to 1; default 0.999)
    lr (float): learning rate (default 1e-4)
    max_steps (int,float): (default 1e5)
    temp (float): annealing factor for the InfoMAX loss (default 1.0)
    perturb (bool): whether to apply perturbations during training
    perturb_fn (function): the perturb function to apply (default micron2.clustering.utils.perturb_x)

  Returns:
    loss_history (list): loss at each step

  """
  # Initialize the queue
  print('Initializing the key queue')
  queue = MoCoQueue(max_queue_len=max_queue_len, zsize=zsize)
  total_enqueued = 0
  for batch in dataset:
    if perturb:
      batch = perturb_fn(batch, crop_size=crop_size)
    key_feat = model.encode_g(batch)
    queue.enqueue(key_feat)
    total_enqueued += batch.shape[0]
    if total_enqueued > max_queue_len:
      break

  optim = tf.keras.optimizers.Adam(learning_rate=lr)
  pbar = tqdm.tqdm(enumerate(dataset), total=int(max_steps))

  print('Start training')
  recent_loss, loss_history = [], []
  prev_recent, prev_overall = 0, 0
  try:
    for i, batch in pbar:
      with tf.GradientTape() as tape:
        if perturb:
          batch = perturb_fn(batch, crop_size=crop_size)

        q_feat = model.encode_g(batch) # The 'active' batch

        with tape.stop_recording():
          key_feat = kmodel.encode_g(batch) # A 'key' batch
        
        batch_size = q_feat.shape[0]
        l = moco_loss(q_feat, key_feat, queue.Q, batch_size=batch_size, temp=temp)

      grads = tape.gradient(l, model.trainable_variables)
      optim.apply_gradients(zip(grads, model.trainable_variables))

      update_key_model(model, kmodel, momentum=momentum)

      lnpy = l.numpy()
      recent_loss.append(lnpy)
      loss_history.append(lnpy)

      if i % 10 == 0:
        mn_recent = np.mean(recent_loss)
        mn_overall = np.mean(loss_history)
        d_recent = prev_recent - mn_recent
        d_overall = prev_overall - mn_overall
        pstr = f'loss s={mn_recent:3.1e}' +\
              f' ds={d_recent:3.1e}' +\
              f' l={mn_overall:3.1e}' +\
              f' dl={d_overall:3.1e}' 
        pbar.set_description(pstr)
        prev_recent = mn_recent
        prev_overall = mn_overall
        recent_loss = []

      # Stash this batch in the training queue
      queue.enqueue(key_feat)
      if i >= max_steps:
        break
  except KeyboardInterrupt:
    print('Caught keyboard interrupt. Ending early.')

  return loss_history