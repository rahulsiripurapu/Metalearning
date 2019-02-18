# -*- coding: utf-8 -*-
"""metalearn2.ipynb

Mini Batch update, Adam, Shuffling batches, SemiLinear.
"""

import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')#Needed for plotting figure on cluster
np.random.seed(1)

def gen_sl_seq(x1,x2):
  w1 = np.random.uniform(-1,1)
  w2 = np.random.uniform(-1,1)
  w3 = np.random.uniform(-1,1)
#   o = np.zeros(len(x1))
#   for i in range(len(x1)):
  o = 0.5*(1+np.tanh(w1*x1+w2*x2+w3))
  return o


def generate_batches(n_batches, batch_size , sequence_length):
  batches = []
  targets = []
  for i in range(n_batches):
    batch = []
    ybatch = []
    for j in range(batch_size):
      x = np.random.uniform(-1,1,(sequence_length,2))
      y = gen_sl_seq(x[:,0],x[:,1])
      y_ = np.roll(y,1)
      x = np.vstack((x.T,y_))
      batch = batch + [x.T]
      ybatch = ybatch + [y]
    batches.append(np.array(batch, dtype=float)) 
    targets.append(np.array(ybatch,dtype=float))
  return batches,targets

tf.reset_default_graph()
tf.set_random_seed(1)
learning_rate = 1e-2
sequence_length = 64
n_batches = 128
b_size = 1
hidden_units = 6
# vec_len = 2
# verbose_freq = 1531
X = tf.placeholder(tf.float32,[None,None,3])
Y = tf.placeholder(tf.float32,[None,None])
batch_size = tf.shape(X)[0]
# Y_hot = tf.one_hot(Y,depth=veclen)
cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
cell_out,cell_state = tf.nn.dynamic_rnn(cell,X,initial_state=init_state)
outputs_flat = tf.reshape(cell_out, [-1, hidden_units])

Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, 1), stddev=0.1))
bout = tf.Variable(tf.zeros(shape=[1])) 
Z = tf.matmul(outputs_flat, Wout) + bout
Y_flat = tf.reshape(Y,[-1,1])
prediction = tf.sigmoid(Z)
loss = tf.losses.mean_squared_error(Y_flat,prediction)
detailed_loss = (Y_flat-prediction)**2
# Y_flat = tf.reshape(Y_hot,[-1,veclen])
# pred = tf.round(Zout)
# predictions = tf.nn.softmax(Z)
# #pred = tf.reshape(pred, [-1, max_len]) # shape: (batch_size, max_len)
hits = tf.equal(tf.round(prediction), Y_flat) 
accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y_flat, logits=Z) 
# loss = tf.reduce_mean(loss)
# learning_rate = tf.train.exponential_decay()
optimizer = tf.train.AdamOptimizer(learning_rate) 
train = optimizer.minimize(loss)

batches,targets = generate_batches(n_batches, b_size , sequence_length)


# %matplotlib inline
session = tf.Session() 
session.run(tf.global_variables_initializer())
examples = np.arange(sequence_length*b_size)
for e in range(1, 10000 + 1):
  state = None
  t = time.time()
  tot_acc = 0
  tot_loss = 0
  indices = np.random.permutation(n_batches)
  for j in range(n_batches):
    if(state is not None):
      feed = {X: batches[indices[j]], Y: targets[indices[j]], init_state: state} 
    else:
      feed = {X: batches[indices[j]], Y: targets[indices[j]]}     
    batch_loss,loss_plot,testy,testpred,state,acc,_ = session.run([loss,detailed_loss,Y_flat,prediction,cell_state,accuracy,train], feed)
    if((e-1)%100==0 and j%50==0):
      print("Plotting")
      plt.figure()
      plt.plot(examples,loss_plot)
      # plt.show()
      plt.savefig(sys.argv[1]+"-epoch-{0}-iter-{1}.png".format(e,j))
      print("{0},{1}".format(testy,testpred))
    tot_acc = tot_acc + acc
    tot_loss = tot_loss + batch_loss
  print((e-1)%100)
  print('Epoch: {0}. Loss: {1}. Time {2}. Accuracy: {3}. avg Loss {4}. avg Acc {5}'.format(e, batch_loss, time.time()-t, acc,tot_loss/n_batches,tot_acc/n_batches))
saver = tf.train.Saver()
save_path = saver.save(session,"data/"+sys.argv[1]+".ckpt")
session.close()
ybatches,ytargets = generate_batches(4, 1 , 500)
ybatches[0][0].shape

session = tf.Session()
saver.restore(session,"data/"+sys.argv[1]+".ckpt")
state = None
t = time.time()
tot_acc = 0
tot_loss = 0
ctr=0
f_loss = np.zeros(2000)
for i in range(500):
  if(state is not None):
      feed = {X: ybatches[0][0][i].reshape(1,1,3), Y: ytargets[0][0][i].reshape(1,1), init_state: state} 
  else:
      feed = {X: ybatches[0][0][i].reshape(1,1,3), Y: ytargets[0][0][i].reshape(1,1)}    
  acc,batch_loss,full_loss,state= session.run([hits,loss,detailed_loss,cell_state], feed)
  print(acc)
  f_loss[i]=full_loss
  tot_loss = tot_loss + batch_loss
  if(acc[0]):
    ctr = ctr + 1
examples = np.arange(1500)
plt.plot(examples,f_loss)
plt.savefig(sys.argv[1]+"_test_output.png")
print('Epoch: {0}. Loss: {1}. Time {2}. Accuracy: {3}. avg Loss {4}. avg Acc {5}.'.format(e, batch_loss, time.time()-t, ctr/1500,tot_loss/1500,tot_acc/1500))

# ybatches,ytargets = generate_batches(1, 16 , 500, index[11:14])

# ybatches[0][0].shape

# session = tf.Session()
# saver.restore(session,"data/"+sys.argv[1]+".ckpt")
# state = None
# t = time.time()
# tot_acc = 0
# tot_loss = 0
# ctr=0
# f_loss = np.zeros(500)
# for i in range(500):
#   if(state is not None):
#       feed = {X: ybatches[0][0][i].reshape(1,1,3), Y: ytargets[0][0][i].reshape(1,1), init_state: state} 
#   else:
#       feed = {X: ybatches[0][0][i].reshape(1,1,3), Y: ytargets[0][0][i].reshape(1,1)}    
#   acc,batch_loss,full_loss,state= session.run([hits,loss,detailed_loss,cell_state], feed)
#   print(acc)
#   f_loss[i]=full_loss
#   tot_loss = tot_loss + batch_loss
#   if(acc[0]):
#     ctr = ctr + 1
# examples = np.arange(500)
# plt.plot(examples,f_loss)
# print('Epoch: {0}. Loss: {1}. Time {2}. Accuracy: {3}. avg Loss {4}. avg Acc {5}.'.format(e, batch_loss, time.time()-t, ctr/500,tot_loss/500,tot_acc/500))
# testout = np.zeros(8000)
# for i in range(len(acc[0])):
#   if(acc[0][i]):
#     testout[i] = 1
#   else:
#     testout[i] = 0
session.close()

# testout = np.reshape(testout,(16,500))

# testout

# x = np.linspace(1,500,500)
# x

# import matplotlib.pyplot as plt
# # %matplotlib inline

# plt.plot(x,testout[7])

