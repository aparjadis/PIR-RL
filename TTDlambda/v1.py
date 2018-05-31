import gym
import numpy
env = gym.make("MountainCar-v0")

Nb_e = 1
h = 10
gamma = 0.99
l = 0.8
G = (h+1)*[0]


def policy(obs):
    if obs[1] < 0:
      a = 0
    else :
      a = 2
    return a

def V(state):
    return 0


for e in range(Nb_e):
    observation = env.reset()
    done = False
    t = 0
    #on garde en mémoire les états et les rewards sur h pas de temps, pour calculer 
    #les lambda returns
    r = (h+1)*[0]
    s = (h+1)*[0]
    
    while not done:
        
      env.render()
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
          print(t," ",Glambda_h)
          #print("shape ",np.array([[s[0][0]],[s[0][1]]]).shape)
          
      
      t += 1
      if done:
          reward += 00
          print("Episode finished after {} timesteps".format(t+1))
          break
    
    #l'episode fini, on fait les h dernières updates 
          r[i] = r[i+1]
          s[i] = s[i+1]
    
    r[h] = reward
    s[h] = observation
    
    
    
    for i in range(1,h-1):
        
        for n in range(i,h+1):
              G[n] = 0
              for j in range(i,n+1):
                  G[n] += gamma**(j-1 - i+1) * r[j]
              G[n] += gamma**n * V(s[n])
        
        
        Glambda = 0
        for j in range(1,h-i):
            Glambda += l**(j-1) * G[j+i]
        Glambda_h = (1-l)*Glambda + l**(h-1-i) * G[h]
        print(t," ",Glambda_h)
        t += 1
    Glambda_h = G[h]
    print(t," ",Glambda_h)
    
env.close()