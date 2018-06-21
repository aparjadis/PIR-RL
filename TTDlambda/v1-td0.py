#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import gym
import tensorflow as tf
import numpy as np
import random as rd
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from batch import MemoryBuffer
env = gym.make("MountainCar-v0")


Nb_episodes = 500
mini_batch_size = 100
#horizon
h = 10
#discount factor
gamma = 0.99
#lambda utilise dans le calcul du lambda return
l = 0.8

tf.reset_default_graph()

G = (h+1)*[0]
replay_memory = MemoryBuffer(1000, (2,))
Learning_Rate = 1e-03
learning_rate = Learning_Rate

x = []
y_0 = []

#policy a evaluer : on accelere dans le sens de la vitesse du vehicule
def policy(obs):
    if obs[1] < 0:
      a = 0
    else :
      a = 2
    return a

    
#input et target du NN
state = tf.placeholder(shape = [None,2], dtype = tf.float32) 
target = tf.placeholder(tf.float32)

#le network
l1 = tf.layers.dense(state, 100, tf.nn.relu)
l2 = tf.layers.dense(l1, 50, tf.nn.relu)
l3 = tf.layers.dense(l2, 10, tf.nn.relu)
NN = tf.layers.dense(l3, 1)

#fonction a minimiser
#loss = tf.losses.mean_squared_error(labels=target, predictions=NN)
loss = tf.reduce_mean((NN - target)**2)


optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#value function
def V(s):
        return sess.run(NN,feed_dict={state: np.reshape(s,(1,2))})

for e in range(Nb_episodes):
#    learning_rate = Learning_Rate/(e+1)
    s_ = env.reset()
    done = False
    t = 0
    err = 0
    #on garde en mémoire les états et les rewards sur h pas de temps, pour calculer 
    #les lambda returns
    r = (h+1)*[0]
    s = (h+1)*[0]
    
    while not done:
        
      #env.render()
      s = s_
      action = policy(s)
      s_, reward, done, info = env.step(action)
      
      if not done:
          targ = reward + gamma*V(s_)
      else:
          targ = reward 
      
      fdict = {state: np.reshape(s,(1,2)), target: targ}
      _, loss_value = sess.run((train, loss),feed_dict=fdict)
      err += loss_value
      
        
    
        
    print("episode ",e," loss value ",err)
    x.append(e)
    if e==0:
        ref = err
    y_0.append(err/ref)
    
plt.plot(x,y_0)
plt.show()


for a in np.linspace(-0.05, 0.05, num=5):
    x_ = np.linspace(-1.2, 0.5, num=100)
    y_ = [V([i,a])[0][0] for i in x_]
    plt.plot(x_,y_)
    plt.show()