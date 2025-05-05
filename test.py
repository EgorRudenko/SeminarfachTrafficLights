import decisionMaking
import numpy as np
from random import randint


cityInput = cityInput = np.array([
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,]])
print(cityInput)



for i in range(100):
    rewards = np.zeros((10000,11//2+1,6+1))
    #rewards = rewards.flatten()
    for o in range(9999):
        rewards = np.zeros((10000,11//2+1,6+1))
        for j in range(11//2+1):
            for k in range(6+1):
                a = randint(0,1)
                #des = decisionMaking.decide((11,6), cityInput, (randint(1,5), randint(1,5)), (a,1,1), randint(1,6000), method="smallReinforce")
                des = decisionMaking.decide((11,6), cityInput, (j, k), (a,1,1), o%101, method="smallReinforce")
                if des[0] != a and des[1] == 1 and des[2] == 1:
                    rewards[o][j][k] += 10
                #else:
                    #rewards[o][j][k] -= 1
                    #rewards[o] += 10
                #if des[1] == 1:
                 #   rewards[o][j][k] += 10
                    #rewards[o] += 10
                #if des[2] == 1:
                 #   rewards[o][j][k] += 10
                    #rewards[o] += 10
        if o % 100 == 0:
            decisionMaking.learn(rewards)
    #rewards = np.reshape(rewards, (101,11//2+1,6+1))
    