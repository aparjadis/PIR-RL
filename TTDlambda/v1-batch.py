from IPython import get_ipython
get_ipython().magic('reset -sf')
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from batch import MemoryBuffer
env = gym.make("MountainCar-v0")


Nb_episodes = 100
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
Learning_Rate = 1e-02
learning_rate = Learning_Rate

x = []
y = []

#policy a evaluer : on accelere dans le sens de la vitesse du vehicule
def policy(obs):
    if obs[1] < 0:
      a = 0
    else :
      a = 2
    return a

def train_on_batch():
#    err = 0
    mBatch = replay_memory.minibatch(mini_batch_size)
    _, loss_value = sess.run((train, loss),feed_dict={state: mBatch[0],target: mBatch[1]})
#    for i in range(mini_batch_size):
#        _, loss_value = sess.run((train, loss),feed_dict={state: np.reshape(mBatch[0][i],(1,2)),target: mBatch[1][i]})
#        err += loss_value
    return loss_value
    
#input et target du NN
state = tf.placeholder(shape = [None,2], dtype = tf.float32) 
target = tf.placeholder(tf.float32)

#le network
l1 = tf.layers.dense(state, 10, tf.nn.relu)
l2 = tf.layers.dense(l1, 500, tf.nn.relu)
l3 = tf.layers.dense(l2, 100, tf.nn.sigmoid)
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
    observation = env.reset()
    done = False
    t = 0
    #on garde en mémoire les états et les rewards sur h pas de temps, pour calculer 
    #les lambda returns
    r = (h+1)*[0]
    s = (h+1)*[0]
    
    while not done:
        
      #env.render()
      action = policy(observation)
      observation, reward, done, info = env.step(action)
      
      for i in range(h):
          r[i] = r[i+1]
          s[i] = s[i+1]
    
      r[h] = reward
      s[h] = observation
    
      if t >= h :
          #calcul des discounted reward a horizon h
          for i in range(1,h+1):
              G[i] = 0
              for j in range(1,i+1):
                  G[i] += gamma**(j-1) * r[j]
              G[i] += gamma**i * V(s[i])
          #calcul du truncated lambda return
          Glambda = 0
          for i in range(1,h):
              Glambda += l**(i-1) * G[i]
          Glambda_h = (1-l)*Glambda + l**(h-1) * G[h]
          replay_memory.append(s[0],Glambda_h)
          
          if e == 1 or e == 4 :
              print(t,Glambda_h)
          

      
      t += 1
      if done:
          #print("Episode finished after ",t," timesteps, for episode number ",e)
          break
    
    #l'episode fini, on fait les h dernières updates 
          r[i] = r[i+1]
          s[i] = s[i+1]
    
    r[h] = reward
    s[h] = observation
    
    for i in range(1,h):
        
        for n in range(i,h+1):
              G[n] = 0
              for j in range(i,n+1):
                  G[n] += gamma**(j-1 - i+1) * r[j]
              if n < h:
                  G[n] += gamma**n * V(s[n])
        
        Glambda = 0
        for j in range(1,h-i):
            Glambda += l**(j-1) * G[j+i]
        Glambda_h = (1-l)*Glambda + l**(h-1-i) * G[h]
        replay_memory.append(s[i],Glambda_h)
        if e == 1 or e == 4:
              print(t,Glambda_h)
        t += 1
        
    Glambda_h = r[h]
    replay_memory.append(s[h],Glambda_h)
    if e == 1 or e == 4:
              print(t,Glambda_h)
    
    err = train_on_batch()
    
    print("episode ",e," loss value ",err)
    x.append(e)
    if e==0:
        ref = err
    y.append(err/ref)
    
plt.plot(x,y)
plt.show()

x = np.linspace(-1.2, 0.5, num=100)
y = [V([i,0])[0][0] for i in x]
plt.plot(x,y)
plt.show()