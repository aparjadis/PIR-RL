import gym
import tensorflow as tf
import numpy as np
import np_fill
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")


Nb_episodes = 100000
mini_batch_size = 32
#horizon
h = 10
#discount factor
gamma = 0.99
#lambda utilise dans le calcul du lambda return
l = 0.8

tf.reset_default_graph()

G = (h+1)*[0]
Learning_Rate = 1e-04
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

#input et target du NN
state = tf.placeholder(shape = [None,2], dtype = tf.float64) 
target = tf.placeholder(tf.float64)
el = tf.placeholder(tf.float32,[8,1])

#le network
l1 = tf.layers.dense(state, 100, tf.nn.relu)
l2 = tf.layers.dense(l1, 50, tf.nn.relu)
l3 = tf.layers.dense(l2, 10, tf.nn.sigmoid)
NN = tf.layers.dense(l3, 1)

#fonction a minimiser
#loss = tf.losses.mean_squared_error(labels=target, predictions=NN)
loss = tf.reduce_mean((NN - target)**2)

W = tf.trainable_variables()
grad_W = tf.gradients(xs=W, ys=loss)

e_old = []
for i in range(len(grad_W)):
    e_old.append(tf.to_double(grad_W[i] > 1000))

e_grad = []
for i in range(len(grad_W)):
    e_grad.append(tf.add(grad_W[i],gamma*l*e_old[i]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_weights = optimizer.apply_gradients(zip(e_grad,W))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#on construit le vecteur de traces d'eligibilite
size_W = 0
for i in range(len(W)):
    for j in range(len(W[i].shape)):
        for k in range(int(W[i].shape[j])):
            size_W += 1
    
#value function
def V(s):
        return sess.run(NN,feed_dict={state: np.reshape(s,(1,2))})

for ep in range(Nb_episodes):
    learning_rate = Learning_Rate/(ep+1)
    s_ = env.reset()
    done = False
    t = 0
    err = 0
    #on garde en mémoire les états et les rewards sur h pas de temps, pour calculer 
    #les lambda returns
    #e = np.zeros((1,size_W))
    
    
    while not done:
        
      #env.render()
      action = policy(s_)   
      s = s_
      s_, reward, done, info = env.step(action)
     
    
      delta = reward + gamma*V(s_) - V(s) 
      
      if t == 0:
          e = sess.run(grad_W,feed_dict={state: np.reshape(s,(1,2)),target: delta})
      else:
          for i in e:
              i *= gamma*l
          grad = sess.run(grad_W,feed_dict={state: np.reshape(s,(1,2)),target: delta})
          for i in range(len(e)):
              e[i] = e[i] + grad[i]
          
      e_grad = delta*e
      _, loss_value = sess.run((update_weights, loss),feed_dict={state: np.reshape(s,(1,2)),target: delta})
      err += loss_value
       
      
#      s1,s2 = sess.run((el,grad_W),feed_dict={state: np.reshape(s,(1,2)),target: delta, el: e})
#      print(s1.shape)
#      print(s2)
      _, loss_value = sess.run((update_weights, loss),feed_dict={state: np.reshape(s,(1,2)),target: delta})
      err += loss_value
      
      t += 1
      if done:
          #print("Episode finished after ",t," timesteps, for episode number ",e)
          break
    
    print("episode ",ep," loss value ",err)
    x.append(ep)
    if ep==0:
        ref = err
    y.append(err/ref)
    
    plt.plot(x,y)