import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")


Nb_episodes = 50
#discount factor
gamma = 0.99

tf.reset_default_graph()

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


optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_weights = optimizer.apply_gradients(zip(grad_W,W))

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
    learning_rate = Learning_Rate/1
    s_ = env.reset()
    done = False
    t = 0
    err = 0
    e_init = 1
    #on garde en mémoire les états et les rewards sur h pas de temps, pour calculer 
    #les lambda returns
    #e = np.zeros((1,size_W))
    
    
    while not done:
        
      #env.render()
      action = policy(s_)   
      s = s_
      s_, reward, done, info = env.step(action)
     
    
      delta = reward + gamma*V(s_) - V(s) 
      
      fdict = {state: np.reshape(s,(1,2)), target: delta}
      _, loss_value = sess.run((update_weights, loss),feed_dict=fdict)
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