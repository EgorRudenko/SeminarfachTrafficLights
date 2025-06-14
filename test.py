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

ah = []
for i in range(1000):
    rewards = []
    #rewards = np.zeros((10000,11//2+1,6+1))
    #rewards = rewards.flatten()
    for o in range(9999):
        for j in range(11//2+1):
            for k in range(6+1):
                a = randint(0,1)
                des = decisionMaking.decide((11,6), cityInput, (j, k), (a,1,1), o%101, method="smallReinforce")
                ah.append(des)
                if des[0] == a and des[1] == 1 and des[2] == 1:
                    rewards.append(10)
                else:
                    rewards.append(0)

        if (o+1) % 100 == 0:
            print(len(ah), len(rewards))
            ah = []
            decisionMaking.learn(rewards)
            rewards = []
    print(len(ah), len(rewards))
    ah = []
    decisionMaking.learn(rewards)
    rewards = []
