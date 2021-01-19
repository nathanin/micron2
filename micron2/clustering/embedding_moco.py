import numpy as np
import tensorflow as tf
import tqdm.auto as tqdm

from .utils import perturb_x

class MoCoQueue:
  def __init__(self, max_queue_len=8):
    self.Q = []
    self.max_queue_len = max_queue_len
  
  def __len__(self):
    return len(self.Q)
      
  def enqueue(self, z):
    self.Q.append(z)
    if len(self) > self.max_queue_len:
      self.dequeue()
  
  def dequeue(self):
    """ Remove the oldest item in queue """
    _ = self.Q.pop(0)
  
  def getqueue(self):
    return tf.concat(self.Q, axis=0)
  

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




def train_moco(dataset, model, kmodel, max_queue_len=64, crop_size=48, 
               momentum=0.999, lr=1e-4, max_steps=1e5, temp=1.0,
               perturb=True):
  """
  Args:
    model, kmodel : identically shaped tf.keras.Model instances, with two sets of parameters
    max_queue_len : number of batches to hold in the key queue
    crop_size : for live perturbations

  """
  # Initialize the queue
  print('Initializing the key queue')
  queue = MoCoQueue(max_queue_len=max_queue_len)
  for batch in dataset:
    if perturb:
      batch = perturb_x(batch, crop_size=crop_size)
    key_feat = model.encode_g(batch)
    queue.enqueue(key_feat)
    if len(queue) == max_queue_len:
      break

  optim = tf.keras.optimizers.Adam(learning_rate=lr)
  pbar = tqdm.tqdm(enumerate(dataset))

  recent_loss, loss_history = [], []
  prev_recent, prev_overall = 0, 0
  for i, batch in pbar:
    with tf.GradientTape() as tape:
      if perturb:
        batch = perturb_x(batch, crop_size=crop_size)

      # q_feat = model.encode_g(perturb_x(batch, crop_size=crop_size)) # The 'active' batch
      q_feat = model.encode_g(batch) # The 'active' batch

      with tape.stop_recording():
        # key_feat = kmodel.encode_g(perturb_x(batch, crop_size=crop_size)) # A 'key' batch
        key_feat = kmodel.encode_g(batch) # A 'key' batch
      
      batch_size = q_feat.shape[0]
      l = moco_loss(q_feat, key_feat, queue.getqueue(), batch_size=batch_size, temp=temp)

    grads = tape.gradient(l, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    update_key_model(model, kmodel, momentum=momentum)

    lnpy = l.numpy()
    recent_loss.append(lnpy)
    loss_history.append(lnpy)

    if i % 100 == 0:
      mn_recent = np.mean(recent_loss)
      mn_overall = np.mean(loss_history)
      d_recent = prev_recent - mn_recent
      d_overall = prev_overall - mn_overall
      pstr = f'loss last-100={mn_recent:3.3e}' +\
             f' d_last-100={d_recent:3.3e}' +\
             f' overall={mn_overall:3.3e}' +\
             f' d_overall={d_overall:3.3e}' 
      pbar.set_description(pstr)
      prev_recent = mn_recent
      prev_overall = mn_overall
      recent_loss = []

    queue.enqueue(key_feat)
    if i >= max_steps:
      break

  return loss_history