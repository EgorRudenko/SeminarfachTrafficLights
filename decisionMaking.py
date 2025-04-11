from random import randint
import numpy as np

rng = np.random  
class ReinforceSmall():
    # configs:
    rewardDecayRate = 0.9
    alpha = 0.01            # learning rate

    # non-config variables
    iterh = []              # iteration history
    ih = []                 # input history
    h1h = []                # hidden layer 1 history
    h2h = []
    h3h = []
    oh = []                 # output layer  history
    ah = []                 # actions history (they are in part random)
    gh = []                 # gradient of cross-entropy history
    def xavier_init(self, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
    def __init__(self, D:int,H:int,H1:int,H2:int,O:int):
        self.model = {
            "W1": self.xavier_init(D, H),  # Input => hidden weights
            "W2": self.xavier_init(H, H1),
            "W3": self.xavier_init(H1, H2),
            "W4": self.xavier_init(H2, O)
        }
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def forwardPropagation(self, streets:list[4], currState, iter:float, pos, dimensions, learningSave:bool = False):
        inputs = []
        for i in streets:
            inputs.append(i)
        for i in currState:
            inputs.append(i)
        inputs.append(iter/1000)
        inputs.append(pos[0]/dimensions[0])
        inputs.append(pos[1]/dimensions[1])
        inputs = np.array(inputs)

        h1 = np.dot(self.model["W1"],inputs)
        h1[h1 < 0] *= 0.1
        h2 = np.dot(self.model["W2"], h1)
        h2[h2 < 0] *= 0.1
        h3 = np.dot(self.model["W3"], h2)
        h3[h3 < 0] *= 0.1
        o = np.dot(self.model["W4"], h3)
        o = list(map(self.sigmoid, o))
        if learningSave:
            self.iterh.append(iter)
            self.ih.append(inputs)
            self.h1h.append(h1)
            self.h2h.append(h2)
            self.h3h.append(h3)
            self.oh.append(o)
        return o
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_reward = 0
        for i in range(1, len(rewards)+1):
            running_reward = running_reward*self.rewardDecayRate + rewards[-i]
            discounted_rewards[-i] = running_reward
        return discounted_rewards

    def back_propagation(self, grads):
        dw1 = np.zeros_like(self.weights["W1"])
        dw2 = np.zeros_like(self.weights["W2"])
        dw3 = np.zeros_like(self.weights["W3"])
        dw4 = np.zeros_like(self.weights["W4"])
        for i in range(len(self.ih)):
            grad = self.gh[self.iterh[i]]
            dw4 += np.dot(self.h3h, grad)
        return {"W1":dw1,"W2":dw2,"W3":dw3,"W4":dw4}

    def learn(self, rewards:list):
        discounted_rewards = self.discount_rewards(rewards)
        grads = np.multiply(discounted_rewards, self.gh)        # actuall gradients (cross-entropy gradient multiplied with )
        self.back_propagation(grads)
        

    def decide(self, city:list, currState, iter, pos, dimensions, learningSave:bool = False):
        s1 = city[pos[0]*2, pos[1]] if pos[1] < dimensions[1] else 0
        s2 = city[pos[0]*2, pos[1]-1] if pos[1]-1 >= 0 else 0
        s3 = city[pos[0]*2-1, pos[1]] if pos[0]*2-1 >= 0 else 0
        s4 = city[pos[0]*2+1, pos[1]] if pos[0]*2+1 < dimensions[0] else 0
        aprob = self.forwardPropagation([s1, s2, s3, s4], currState, iter, pos, dimensions, True)
        actions = list(map(lambda a: 1 if rng.uniform() < a else 0, aprob))
        if learningSave:
            gradient_of_cross_entropy = []
            for i in range(3):
                gradient_of_cross_entropy.append(actions[i]-aprob[i])
            self.gh.append(gradient_of_cross_entropy)

        return actions



smallReinforce = ReinforceSmall(10, 20, 10, 5, 3)

def learn(rewards, model="smallReinforce"):
    global smallReinforce
    match model:
        case "smallReinforce":
            smallReinforce.learn(rewards)

def decide(dimensions, city, pos, currState, iter:int, method:str="randomD"):
    global smallReinforce
    match method:
        case "random":
            return random()
        case "oppositeWithTurns":
            return alwaysChangeWithTurns(currState)
        case "smallReinforce":  
            return smallReinforce.decide(city, currState, iter, pos, dimensions, True)

def random():
    return [randint(0,1),randint(0,1),randint(0,1)]

def alwaysChangeWithTurns(currState):
    return (not currState[0], 1, 1)
