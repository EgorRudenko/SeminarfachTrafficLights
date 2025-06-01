from random import randint, seed
import numpy as np
from scipy import stats


seed(2)
np.random.seed(2)

rng = np.random.default_rng(seed = 1) 
class ReinforceSmall():
    # configs:
    rewardDecayRate = 0.9
    alpha = 0.0001          # learning rate
    decay_rmsprop = 0.9
    mem = 10000
    bias_alpha = 0.0001
    reluCoeff = 0.01
    exploration = 1.5
    gradClip = 1

    rmsprop_cache = []
    # non-config variables
    def init(self):
        self.iterh = []              # iteration history
        self.ih = []                 # input history
        self.h1h = []                # hidden layer 1 history
        self.h2h = []
        self.h3h = []
        self.oh = []                 # output layer  history
        self.ah = []                 # actions history (they are in part random)
        self.gh = []                 # gradient of cross-entropy history
        self.gh1 = []
        self.descisionsMade = 0
        self.sumOfDescisions = np.array([0.0,0.0,0.0])
        self.sumOfDescisions1 = np.array([0.0,0.0,0.0])
    def xavier_init(self, fan_in, fan_out):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
    def he_init(self, fan_in, fan_out):
        std = np.sqrt(2/fan_in)
        return rng.normal(0, std, size=(fan_out, fan_in))
    def __init__(self, D:int,H:int,H1:int,H2:int,O:int):
        self.init()
        self.D = D
        self.H = H
        self. H1 = H1
        self.H2 = H2
        self.O = O
        self.model = {
            "W1": self.he_init(D, H),  # Input => hidden weights
            "W2": self.he_init(H, H1),
            "W3": self.he_init(H1, H2),
            "W4": self.xavier_init(H2, O)
        }
        self.biases = [
            np.zeros(D),
            np.zeros(H),
            np.zeros(H1),
            np.zeros(H2),
            np.zeros(O)
        ]
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def forwardPropagation(self, streets:list[4], currState, iter:float, pos, dimensions, learningSave:bool = False):
        inputs = []
        for i in streets:
            inputs.append(i)
        for i in currState:
            inputs.append(i)
        inputs.append((self.sigmoid(iter/500)-0.5)*2)
        inputs.append(pos[0]/dimensions[0])
        inputs.append(pos[1]/dimensions[1])
        inputs = np.array(inputs)
        #print(inputs)
        h1 = np.dot(self.model["W1"],inputs+self.biases[0])+self.biases[1]
        h1[h1 < 0] *= self.reluCoeff
        h2 = np.dot(self.model["W2"], h1)+self.biases[2]
        h2[h2 < 0] *= self.reluCoeff
        h3 = np.dot(self.model["W3"], h2)+self.biases[3]
        h3[h3 < 0] *= self.reluCoeff
        o = np.dot(self.model["W4"], h3)+self.biases[4]
        o = list(map(self.sigmoid, o))
        if learningSave:
            self.iterh.append(iter)
            self.ih.append(inputs+self.biases[0])
            self.h1h.append(h1+self.biases[1])
            self.h2h.append(h2+self.biases[2])
            self.h3h.append(h3+self.biases[3])
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
        dw1 = np.zeros_like(self.model["W1"])
        dw2 = np.zeros_like(self.model["W2"])
        dw3 = np.zeros_like(self.model["W3"])
        dw4 = np.zeros_like(self.model["W4"])
        si = np.zeros((1,self.D))
        sh = np.zeros((1,self.H))
        sh1 = np.zeros((1,self.H1))
        sh2 = np.zeros((1,self.H2))
        so = np.zeros((1,self.O))

        #grads = grads.reshape(int(grads.size/3),3)
        for i in range(len(self.ih)):
            if grads[i][0] != 0 and self.oh[i][2] > 0.99:
                pass
                #print(self.ih[i], grads[i], self.oh[i])
            grad = np.array([grads[i]])
            so += grad
            dw4 += np.dot(grad.T, np.array([self.h3h[i]]))

            h3 = np.dot(grad, self.model["W4"])
            h3[h3<0] *= self.reluCoeff
            sh2 += h3
            dw3 += np.dot(h3.T, np.array([self.h2h[i]]))

            h2 = np.dot(h3, self.model["W3"])
            h2[h2<0] *= self.reluCoeff
            sh1 += h2
            dw2 += np.dot(h2.T, np.array([self.h1h[i]]))

            h1 = np.dot(h2, self.model["W2"])
            sh += h1
            h1[h1<0] *= self.reluCoeff
            dw1 += np.dot(h1.T, np.array([self.ih[i]]))
        dw1 /= len(self.ih)
        dw2 /= len(self.ih)
        dw3 /= len(self.ih)
        dw4 /= len(self.ih)

        si /= len(self.ih)
        sh /= len(self.ih)
        sh1 /= len(self.ih)
        sh2 /= len(self.ih)
        so /= len(self.ih)
        return {"W1":dw1,"W2":dw2,"W3":dw3,"W4":dw4}, [si, sh, sh1, sh2, so]

    def learn(self, rewards:list, mode):
        #print(rewards.shape)
        if mode == "vocal":
            print(f"average certainty: {2*self.sumOfDescisions / self.descisionsMade}")
            print(f"average descision: {self.sumOfDescisions1 / self.descisionsMade}")
            print(f"Max reward: {np.max(rewards)}; Min reward: {np.min(rewards)}; Average reward: {np.average(np.trim_zeros(rewards))}")
            print("learining...")
        if np.any(rewards):
            rewards -= np.mean(rewards) 
            rewards /= np.std(rewards)
        print(len(self.gh1), len(rewards))
        for i in range(len(self.gh1)-1):
            self.gh1[i] = self.gh1[i]*rewards[i]
        #for i in range(len(self.gh)):
        #    for j in range(len(self.gh[i])):
        #        for k in range(len(self.gh[i][j])):
        #            self.gh[i][j][k] = self.gh[i][j][k]*rewards[i][j][k]
        if mode == "vocal":
            print("prepare to do gradients...")
        deltas, bias_deltas = self.back_propagation(self.gh1)
        for k, v in deltas.items():
            #deltas[k] /= np.std(deltas[k])
            deltas[k][deltas[k] > self.gradClip] = self.gradClip
            deltas[k][deltas[k] < -self.gradClip] = -self.gradClip
        
        for i in range(len(bias_deltas)):
            self.biases[i] += bias_deltas[i].flatten()*self.bias_alpha
            

        if mode == "vocal":
            #print("Gradients: ", deltas)
            #print("Weights: ", self.model)
            print("gradients done, applying gradients...")
        # apply gradients with rmsprop
        for k, v in self.model.items():
            g = deltas[k]
            self.rmsprop_cache[k] = (
                self.rmsprop_cache[k] * self.decay_rmsprop + (1-self.decay_rmsprop)*(g**2)
            ) 
            self.model[k] += self.alpha * g/(np.sqrt(self.rmsprop_cache[k])+0.0000001)
            #self.model[k] += self.alpha * deltas[k]
        if mode == "vocal":
            print("gradients applied")
        self.init()


    def decide(self, city:list, currState, iter, pos, dimensions, learningSave:bool = False):
        if len(self.gh) == 0:
            self.gh = np.zeros((self.mem,dimensions[0]//2+1, dimensions[1]+1, 3))
        s1 = city[pos[0]*2, pos[1]] if pos[1] < dimensions[1] else 0
        s2 = city[pos[0]*2, pos[1]-1] if pos[1]-1 >= 0 else 0
        s3 = city[pos[0]*2-1, pos[1]] if pos[0]*2-1 >= 0 else 0
        s4 = city[pos[0]*2+1, pos[1]] if pos[0]*2+1 < dimensions[0] else 0
        aprob = self.forwardPropagation([s1, s2, s3, s4], currState, iter, pos, dimensions, learningSave)
        #actions = list(map(lambda a: 1 if stats.truncnorm.rvs(-1, 1,loc = 0.5, scale = 0.5, size = 1) < a else 0, aprob))
        exploration = self.exploration
        actions = list(map(lambda a: 1 if rng.uniform() < a/exploration + (exploration - 1)/(exploration*2) else 0, aprob))
        self.ah.append(actions)
        self.sumOfDescisions += np.abs(np.array([0.5,0.5,0.5]) - aprob)
        self.sumOfDescisions1 += aprob
        self.descisionsMade += 1
        if learningSave:
            gradient_of_cross_entropy = []
            for i in range(3):
                gradient_of_cross_entropy.append(actions[i]-aprob[i])
            #print(self.gh)
            #print("-------------------------------", gradient_of_cross_entropy, actions, aprob)
            self.gh1.append(np.array(gradient_of_cross_entropy))
            self.gh[iter][pos[0]][pos[1]] = gradient_of_cross_entropy

        return actions



smallReinforce = ReinforceSmall(10, 30, 10, 5, 3)

def learn(rewards, mode = "vocal", model="smallReinforce"):
    global smallReinforce
    match model:
        case "smallReinforce":
            smallReinforce.learn(rewards, mode)

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
