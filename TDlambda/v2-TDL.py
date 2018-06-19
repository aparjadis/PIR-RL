import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")

##S'arrete apres x pas de temps (avant divergence sur un episode)

Nb_episodes = 1
#discount factor
gamma = 0.99
#lambda utilise dans le calcul du lambda return
l = 0.8

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
state = tf.placeholder(shape = [None,2], dtype=tf.float32)
delta = tf.placeholder(shape = [], dtype=tf.float32)


#le network
l1 = tf.layers.dense(state, 100, tf.nn.relu)
l2 = tf.layers.dense(l1, 50, tf.nn.relu)
l3 = tf.layers.dense(l2, 10, tf.nn.relu)
NN = tf.layers.dense(l3, 1)

#fonction a minimiser
#loss = tf.losses.mean_squared_error(labels=target, predictions=NN)
#loss = tf.reduce_mean((NN - delta)**2)

#On recupere les poids et le gradient de la fonction loss par rapport aux poids
W = tf.trainable_variables()
w = W[::2]
b = W[1::2]
#grad_w = tf.gradients(xs=w, ys=loss)
#grad_b = tf.gradients(xs=b, ys=loss)
grad = tf.gradients(xs=W, ys=NN)

#On initialise E_t-1, de même forme que W
#e_old_w = [tf.Variable([[0. for j in range(grad_w[i].shape[1])] for k in range(grad_w[i].shape[0])],dtype=tf.float64) for i in range(len(grad_w))]
#e_old_b = [tf.Variable([0. for j in range(grad_b[i].shape[0])],dtype=tf.float64) for i in range(len(grad_b))]
#for i in range(len(grad_W)):
#    e_old[i] = tf.to_double(grad_W[i] > 1000)
#    e_old[i] = tf.cond(init_el_traces,lambda : e_init(i),lambda: tf.identity(e_old[i]))

len_grad = len(grad)

e = [tf.Variable(tf.zeros(grad[i].shape)) for i in range(len_grad)]

#On construit E_t à partir du gradient et de E_t-1
#e_grad_w = [tf.add(grad_w[i],gamma*l*e_old_w[i]) for i in range(len(grad_w))]
#e_grad_b = [tf.add(grad_b[i],gamma*l*e_old_b[i]) for i in range(len(grad_b))]
e_new = [tf.add(gamma*l*e[i],grad[i]) for i in range(len_grad)]
#for i in range(len(grad_W)):
#    e_grad.append(tf.add(grad_W[i],gamma*l*e_old[i]))

#E_t-1 <-- E_t
#for i in range(len(grad_W)):
#    e_old[i] = e_grad[i] 
#eligibility_step = tf.assign(e_old_w[0], e_grad_w[0])
w_diff = [tf.scalar_mul(delta, e_new[i]) for i in range(len_grad)]

update_e = [tf.assign_add(e[i], e_new[i]) for i in range(len_grad)]

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#update_weights_w = optimizer.apply_gradients(zip(e_grad_w,w))
#update_weights_b = optimizer.apply_gradients(zip(e_grad_b,b))
update_weights = optimizer.apply_gradients(zip(w_diff,W))

updates = update_e + [update_weights] + [delta] + [e_new] + [grad]

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

    
#value function
def V(s):
        return sess.run(NN,feed_dict={state: np.reshape(s,(1,2))})

for ep in range(Nb_episodes):
    learning_rate = Learning_Rate
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
     
    
      Delta = reward + gamma*V(s_) - V(s) 
      
#      if t == 0:
#          fdict = {state: np.reshape(s,(1,2)),target: delta,init_el_traces: e_init}
#          e = sess.run(grad_W,feed_dict=fdict)
#          e_init = 0
#      else:
#          for i in e:
#              i *= gamma*l
#          grad = sess.run(grad_W,feed_dict={state: np.reshape(s,(1,2)),target: delta})
#          for i in range(len(e)):
#              e[i] = e[i] + grad[i]
          
      
      fdict = {state: np.reshape(s,(1,2)), delta: Delta[0][0]}
      res = sess.run(updates,feed_dict=fdict)
      err += res[-3]
      
#      if t > 2:
#          
#          plt.plot(e1[0])
#          plt.show()
       
      
#      s1,s2 = sess.run((el,grad_W),feed_dict={state: np.reshape(s,(1,2)),target: delta, el: e})
#      print(s1.shape)
#      print(s2)
      #_, loss_value = sess.run((update_weights, loss),feed_dict={state: np.reshape(s,(1,2)),target: delta})
      #err += loss_value
      
      if t == 20:
          break
      
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